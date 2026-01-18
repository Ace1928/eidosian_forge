from __future__ import annotations
import itertools
import os
import re
import warnings
from collections import UserDict
from typing import TYPE_CHECKING, Any
import numpy as np
import spglib
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
@staticmethod
def get_all_possible_basis_functions(structure: Structure, potcar_symbols: list, address_basis_file_min: str | None=None, address_basis_file_max: str | None=None):
    """
        Args:
            structure: Structure object
            potcar_symbols: list of the potcar symbols
            address_basis_file_min: path to file with the minimum required basis by the POTCAR
            address_basis_file_max: path to file with the largest possible basis of the POTCAR.

        Returns:
            list[dict]: Can be used to create new Lobsterin objects in
                standard_calculations_from_vasp_files as dict_for_basis
        """
    max_basis = Lobsterin.get_basis(structure=structure, potcar_symbols=potcar_symbols, address_basis_file=address_basis_file_max or f'{MODULE_DIR}/lobster_basis/BASIS_PBE_54_max.yaml')
    min_basis = Lobsterin.get_basis(structure=structure, potcar_symbols=potcar_symbols, address_basis_file=address_basis_file_min or f'{MODULE_DIR}/lobster_basis/BASIS_PBE_54_min.yaml')
    all_basis = get_all_possible_basis_combinations(min_basis=min_basis, max_basis=max_basis)
    list_basis_dict = []
    for basis in all_basis:
        basis_dict = {}
        for elba in basis:
            basplit = elba.split()
            basis_dict[basplit[0]] = ' '.join(basplit[1:])
        list_basis_dict.append(basis_dict)
    return list_basis_dict