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
def check_eq_forces(self):
    """Check maximum size of forces in the equilibrium structure."""
    name = f'{self.name}.eq'
    feq_av = self.cache[name]['forces']
    fmin = feq_av.max()
    fmax = feq_av.min()
    i_min = np.where(feq_av == fmin)
    i_max = np.where(feq_av == fmax)
    return (fmin, fmax, i_min, i_max)