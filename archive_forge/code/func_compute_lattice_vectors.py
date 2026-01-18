from math import pi, sqrt
import warnings
from pathlib import Path
import numpy as np
import numpy.linalg as la
import numpy.fft as fft
import ase
import ase.units as units
from ase.parallel import world
from ase.dft import monkhorst_pack
from ase.io.trajectory import Trajectory
from ase.utils.filecache import MultiFileJSONCache
def compute_lattice_vectors(self):
    """Return lattice vectors for cells in the supercell."""
    R_cN = np.indices(self.supercell).reshape(3, -1)
    N_c = np.array(self.supercell)[:, np.newaxis]
    if self.offset == 0:
        R_cN += N_c // 2
        R_cN %= N_c
    R_cN -= N_c // 2
    return R_cN