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
@classmethod
def from_cif_file(cls, cif_file: str, source: str='', comment: str='') -> Self:
    """
        Create Header object from cif_file.

        Args:
            cif_file: cif_file path and name
            source: User supplied identifier, i.e. for Materials Project this
                would be the material ID number
            comment: User comment that goes in header

        Returns:
            Header Object
        """
    parser = CifParser(cif_file)
    structure = parser.parse_structures(primitive=True)[0]
    return cls(structure, source, comment)