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
def from_structure_with_site_properties_Ucif(cls, structure: Structure, temperature: float | None=None) -> Self:
    """Will create this object with the help of a structure with site properties.

        Args:
            structure: Structure object including U11_cif, U22_cif, U33_cif, U23_cif, U13_cif, U12_cif as site
            properties
            temperature: temperature for Ucif data

        Returns:
            ThermalDisplacementMatrices
        """
    Ucif_matrix = []
    for site in structure:
        Ucif_matrix.append([site.properties[f'U{idx}_cif'] for idx in (11, 22, 33, 23, 13, 12)])
    return cls.from_Ucif(Ucif_matrix, structure, temperature=temperature)