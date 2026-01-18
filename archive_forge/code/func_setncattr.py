import sys
import h5py
import numpy as np
from . import core
def setncattr(self, name, value):
    """Set a netCDF4 attribute."""
    self.attrs[name] = value