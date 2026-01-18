import numpy as np
import warnings
from ase.md.nvtberendsen import NVTBerendsen
import ase.units as units
def set_compressibility(self, *, compressibility_au):
    self.compressibility = compressibility_au