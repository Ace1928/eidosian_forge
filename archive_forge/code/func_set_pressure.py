import numpy as np
import warnings
from ase.md.nvtberendsen import NVTBerendsen
import ase.units as units
def set_pressure(self, pressure=None, *, pressure_au=None, pressure_bar=None):
    self.pressure = self._process_pressure(pressure, pressure_bar, pressure_au)