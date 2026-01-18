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
def read_born_charges(self, name=None, neutrality=True):
    """Read Born charges and dieletric tensor from JSON file.

        The charge neutrality sum-rule::

                   _ _
                   \\    a
                    )  Z   = 0
                   /__  ij
                    a

        Parameters:

        neutrality: bool
            Restore charge neutrality condition on calculated Born effective
            charges.

        """
    if name is None:
        key = '%s.born' % self.name
    else:
        key = name
    Z_avv, eps_vv = self.cache[key]
    if neutrality:
        Z_mean = Z_avv.sum(0) / len(Z_avv)
        Z_avv -= Z_mean
    self.Z_avv = Z_avv[self.indices]
    self.eps_vv = eps_vv