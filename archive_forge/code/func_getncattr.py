import sys
import h5py
import numpy as np
from . import core
def getncattr(self, name):
    """Retrieve a netCDF4 attribute."""
    return self.attrs[name]