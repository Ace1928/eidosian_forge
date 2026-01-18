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
@staticmethod
def get_reduced_matrix(thermal_displacement: ArrayLike[ArrayLike]) -> np.ndarray[np.ndarray]:
    """Transfers the full matrix to reduced matrix (order of reduced matrix U11, U22, U33, U23, U13, U12).

        Args:
            thermal_displacement: 2d numpy array, first dimension are the atoms

        Returns:
            3d numpy array including thermal displacements, first dimensions are the atoms
        """
    reduced_matrix = np.zeros((len(thermal_displacement), 6))
    for imat, mat in enumerate(thermal_displacement):
        reduced_matrix[imat][0] = mat[0][0]
        reduced_matrix[imat][1] = mat[1][1]
        reduced_matrix[imat][2] = mat[2][2]
        reduced_matrix[imat][3] = mat[1][2]
        reduced_matrix[imat][4] = mat[0][2]
        reduced_matrix[imat][5] = mat[0][1]
    return reduced_matrix