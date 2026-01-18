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
def get_atom_forces(self) -> np.ndarray:
    """Return the force on atoms in material system.

        Returns:
            np.ndarray: Forces acting on individual atoms of shape=(num_atoms*3,)
        """
    forces = []
    aim_content = 'Force'.upper()
    aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
    for line in self.strs_lst[aim_idx + 1:aim_idx + self.num_atoms + 1]:
        forces.append([float(val) for val in line.split()[1:4]])
    return -np.array(forces).reshape(-1)