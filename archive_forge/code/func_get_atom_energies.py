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
def get_atom_energies(self) -> np.ndarray | None:
    """Return the energies of individual atoms in material system.

        Returns:
            np.ndarray | None : The energies of individual atoms within the material system.

        Description:
            When turn on `ENERGY DEPOSITION`, PWmat will output energy per atom.
        """
    energies = []
    aim_content = 'Atomic-Energy, '.upper()
    aim_idxs = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)
    if len(aim_idxs) == 0:
        return None
    aim_idx = aim_idxs[0]
    for tmp_str in self.strs_lst[aim_idx + 1:aim_idx + self.num_atoms + 1]:
        '\n            Atomic-Energy, Etot(eV),E_nonloc(eV),Q_atom:dE(eV)=  -0.1281163115E+06\n            14   0.6022241483E+03    0.2413350871E+02    0.3710442365E+01\n            '
        energies.append(float(tmp_str.split()[1]))
    return np.array(energies)