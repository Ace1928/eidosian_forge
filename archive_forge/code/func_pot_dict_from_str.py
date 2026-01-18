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
def pot_dict_from_str(pot_data):
    """
        Creates atomic symbol/potential number dictionary
        forward and reverse.

        Args:
            pot_data: potential data in string format

        Returns:
            forward and reverse atom symbol and potential number dictionaries.
        """
    pot_dict = {}
    pot_dict_reverse = {}
    begin = 0
    ln = -1
    for line in pot_data.split('\n'):
        try:
            if begin == 0 and line.split()[0] == '0':
                begin += 1
                ln = 0
            if begin == 1:
                ln += 1
            if ln > 0:
                atom = line.split()[2]
                index = int(line.split()[0])
                pot_dict[atom] = index
                pot_dict_reverse[index] = atom
        except (ValueError, IndexError):
            pass
    return (pot_dict, pot_dict_reverse)