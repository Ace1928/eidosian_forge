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
def header_string_from_file(filename: str='feff.inp'):
    """
        Reads Header string from either a HEADER file or feff.inp file
        Will also read a header from a non-pymatgen generated feff.inp file.

        Args:
            filename: File name containing the Header data.

        Returns:
            Reads header string.
        """
    with zopen(filename, mode='r') as fobject:
        f = fobject.readlines()
        feff_header_str = []
        ln = 0
        try:
            feff_pmg = f[0].find('pymatgen')
            if feff_pmg == -1:
                feff_pmg = False
        except IndexError:
            feff_pmg = False
        if feff_pmg:
            n_sites = int(f[8].split()[2])
            for line in f:
                ln += 1
                if ln <= n_sites + 9:
                    feff_header_str.append(line)
        else:
            end = 0
            for line in f:
                if (line[0] == '*' or line[0] == 'T') and end == 0:
                    feff_header_str.append(line.replace('\r', ''))
                else:
                    end = 1
    return ''.join(feff_header_str)