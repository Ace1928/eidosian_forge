import numpy as np
import warnings
from ase.md.nvtberendsen import NVTBerendsen
import ase.units as units
def set_taup(self, taup):
    self.taup = taup