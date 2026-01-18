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
def get_virial(self) -> np.ndarray | None:
    """Return the virial tensor of material system.

        Returns:
            np.ndarray | None: Virial tensor of shape=(9,)
        """
    virial_tensor = []
    aim_content = 'LATTICE'
    aim_idx = ListLocator.locate_all_lines(strs_lst=self.strs_lst, content=aim_content)[0]
    for tmp_idx in [aim_idx + 1, aim_idx + 2, aim_idx + 3]:
        tmp_strs_lst = self.strs_lst[tmp_idx].split()
        tmp_aim_row_lst = ListLocator.locate_all_lines(strs_lst=tmp_strs_lst, content='STRESS')
        if len(tmp_aim_row_lst) == 0:
            return None
    for tmp_idx in [aim_idx + 1, aim_idx + 2, aim_idx + 3]:
        tmp_str_lst = self.strs_lst[tmp_idx].split()[-3:]
        virial_tensor.append(float(tmp_str_lst[0]))
        virial_tensor.append(float(tmp_str_lst[1]))
        virial_tensor.append(float(tmp_str_lst[2]))
    return np.array(virial_tensor)