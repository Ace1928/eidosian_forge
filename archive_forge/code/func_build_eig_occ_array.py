import os
import copy
import subprocess
from math import pi, sqrt
import pathlib
from typing import Union, Optional, List, Set, Dict, Any
import warnings
import numpy as np
from ase.cell import Cell
from ase.outputs import Properties, all_outputs
from ase.utils import jsonable
from ase.calculators.abc import GetPropertiesMixin
def build_eig_occ_array(self, getter):
    nspins = self.get_number_of_spins()
    nkpts = len(self.get_ibz_k_points())
    nbands = self.get_number_of_bands()
    arr = np.zeros((nspins, nkpts, nbands))
    for s in range(nspins):
        for k in range(nkpts):
            arr[s, k, :] = getter(spin=s, kpt=k)
    return arr