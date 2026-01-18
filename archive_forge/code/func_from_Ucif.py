from __future__ import annotations
import re
from functools import partial
from typing import TYPE_CHECKING
import numpy as np
from monty.json import MSONable
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifFile, CifParser, CifWriter, str2float
from pymatgen.symmetry.groups import SYMM_DATA
from pymatgen.util.due import Doi, due
@classmethod
def from_Ucif(cls, thermal_displacement_matrix_cif: ArrayLike[ArrayLike], structure: Structure, temperature: float | None=None) -> Self:
    """Starting from a numpy array, it will convert Ucif values into Ucart values and initialize the class.

        Args:
            thermal_displacement_matrix_cif: np.array,
                first dimension are the atoms,
                then reduced form of thermal displacement matrix will follow
                Order as above: U11, U22, U33, U23, U13, U12
            structure: Structure object
            temperature: float
                Corresponding temperature

        Returns:
            ThermalDisplacementMatrices
        """
    thermal_displacement_matrix_cif_matrix_form = ThermalDisplacementMatrices.get_full_matrix(thermal_displacement_matrix_cif)
    A = structure.lattice.matrix.T
    N = np.diag([np.linalg.norm(x) for x in np.linalg.inv(A)])
    Ucart = []
    for mat in thermal_displacement_matrix_cif_matrix_form:
        mat_ustar = np.dot(np.dot(N, mat), N.T)
        mat_ucart = np.dot(np.dot(A, mat_ustar), A.T)
        Ucart.append(mat_ucart)
    thermal_displacement_matrix_cart = ThermalDisplacementMatrices.get_reduced_matrix(np.array(Ucart))
    return cls(thermal_displacement_matrix_cart=thermal_displacement_matrix_cart, thermal_displacement_matrix_cif=thermal_displacement_matrix_cif, structure=structure, temperature=temperature)