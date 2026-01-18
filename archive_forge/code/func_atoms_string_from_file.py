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
@staticmethod
def atoms_string_from_file(filename):
    """
        Reads atomic shells from file such as feff.inp or ATOMS file
        The lines are arranged as follows:

        x y z   ipot    Atom Symbol   Distance   Number

        with distance being the shell radius and ipot an integer identifying
        the potential used.

        Args:
            filename: File name containing atomic coord data.

        Returns:
            Atoms string.
        """
    with zopen(filename, mode='rt') as fobject:
        f = fobject.readlines()
        coords = 0
        atoms_str = []
        for line in f:
            if coords == 0:
                find_atoms = line.find('ATOMS')
                if find_atoms >= 0:
                    coords = 1
            if coords == 1 and 'END' not in line:
                atoms_str.append(line.replace('\r', ''))
    return ''.join(atoms_str)