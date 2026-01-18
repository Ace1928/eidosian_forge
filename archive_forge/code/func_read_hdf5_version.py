import os
import os.path
import warnings
from ..base import CommandLine
def read_hdf5_version(s):
    if 'HDF5' in s:
        return s.split(':')[1].strip()
    return None