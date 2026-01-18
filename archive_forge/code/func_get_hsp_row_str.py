from __future__ import annotations
import linecache
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.kpath import KPathSeek
def get_hsp_row_str(label: str, index: int, coordinate: float) -> str:
    """
            Return string containing name, index, coordinate of the certain high symmetry point
            in HIGH_SYMMETRY_POINTS format.

            Args:
                label (str): Name of the high symmetry point.
                index (int): Index of the high symmetry point.
                coordinate (float): Coordinate in bandstructure of the high symmetry point.

            Returns:
                str: String containing name, index, coordinate of the certain high symmetry point
                    in HIGH_SYMMETRY_POINTS format.
            """
    if label == 'GAMMA':
        return f'G            {index:>4d}         {coordinate:>.6f}\n'
    return f'{label}            {index:>4d}         {coordinate:>.6f}\n'