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
def define_offset(self):
    if not self.center_refcell:
        self.offset = 0
    else:
        N_c = self.supercell
        self.offset = N_c[0] // 2 * (N_c[1] * N_c[2]) + N_c[1] // 2 * N_c[2] + N_c[2] // 2
    return self.offset