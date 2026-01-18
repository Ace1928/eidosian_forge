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
def from_cif_P1(filename: str) -> list[ThermalDisplacementMatrices]:
    """Reads a cif with P1 symmetry including positions and ADPs.
        Currently, no check of symmetry is performed as CifParser methods cannot be easily reused.

        Args:
            filename: Filename of the CIF.

        Returns:
            ThermalDisplacementMatrices
        """
    cif = CifFile.from_file(filename)
    thermals = []
    for data in cif.data.values():
        lattice = CifParser.get_lattice_no_exception(data)
        all_coords = []
        all_species = []
        for idx in range(len(data['_atom_site_label'])):
            try:
                symbol = CifParser(filename)._parse_symbol(data['_atom_site_type_symbol'][idx])
            except KeyError:
                symbol = CifParser(filename)._parse_symbol(data['_atom_site_label'][idx])
            if not symbol:
                continue
            all_species.append(symbol)
            x = str2float(data['_atom_site_fract_x'][idx])
            y = str2float(data['_atom_site_fract_y'][idx])
            z = str2float(data['_atom_site_fract_z'][idx])
            all_coords.append([x, y, z])
        thermals_Ucif = [[str2float(data['_atom_site_aniso_U_11'][idx]), str2float(data['_atom_site_aniso_U_22'][idx]), str2float(data['_atom_site_aniso_U_33'][idx]), str2float(data['_atom_site_aniso_U_23'][idx]), str2float(data['_atom_site_aniso_U_13'][idx]), str2float(data['_atom_site_aniso_U_12'][idx])] for idx in range(len(data['_atom_site_aniso_label']))]
        struct = Structure(lattice, all_species, all_coords)
        thermal = ThermalDisplacementMatrices.from_Ucif(thermal_displacement_matrix_cif=thermals_Ucif, structure=struct, temperature=None)
        thermals.append(thermal)
    return thermals