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
def get_e_tot(self) -> np.ndarray:
    """Return the total energy of structure.

        Returns:
            np.ndarray: The total energy of the material system.
        """
    strs_lst = self.strs_lst[0].split(',')
    aim_index = ListLocator.locate_all_lines(strs_lst=strs_lst, content='EK (EV) =')[0]
    return np.array([float(strs_lst[aim_index].split()[3].strip())])