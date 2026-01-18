import sys
import h5py
import numpy as np
from . import core
def ncattrs(self):
    """Return netCDF4 attribute names."""
    return list(self.attrs)