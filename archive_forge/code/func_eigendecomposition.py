import os
from typing import Any, Union
import numpy as np
from ase import Atoms
from ase.io.jsonio import read_json, write_json
from ase.parallel import world, parprint
def eigendecomposition(self, omega, seed=0):
    u, s, v = np.linalg.svd(omega)
    generator = np.random.RandomState(seed)
    return (s, v.T, generator)