import numpy as np
import warnings
from ase.md.nvtberendsen import NVTBerendsen
import ase.units as units
def get_timestep(self):
    return self.dt