from collections.abc import (
import os
import posixpath
import numpy as np
from .._objects import phil, with_phil
from .. import h5d, h5i, h5r, h5p, h5f, h5t, h5s
from .compat import fspath, filename_encode
def is_hdf5(fname):
    """ Determine if a file is valid HDF5 (False if it doesn't exist). """
    with phil:
        fname = os.path.abspath(fspath(fname))
        if os.path.isfile(fname):
            return h5f.is_hdf5(filename_encode(fname))
        return False