from __future__ import annotations
import re
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from tabulate import tabulate
from pymatgen.core import Element, Lattice, Molecule, Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.core import ParseError
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.io_utils import clean_lines
from pymatgen.util.string import str_delimited
def get_atom_map(structure, absorbing_atom=None):
    """
    Returns a dict that maps each atomic symbol to a unique integer starting
    from 1.

    Args:
        structure (Structure)
        absorbing_atom (str): symbol

    Returns:
        dict
    """
    unique_pot_atoms = sorted({site.specie.symbol for site in structure})
    if absorbing_atom and len(structure.indices_from_symbol(absorbing_atom)) == 1:
        unique_pot_atoms.remove(absorbing_atom)
    atom_map = {}
    for i, atom in enumerate(unique_pot_atoms, start=1):
        atom_map[atom] = i
    return atom_map