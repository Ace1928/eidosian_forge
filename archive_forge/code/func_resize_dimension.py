import os.path
import warnings
import weakref
from collections import ChainMap, Counter, OrderedDict, defaultdict
from collections.abc import Mapping
import h5py
import numpy as np
from packaging import version
from . import __version__
from .attrs import Attributes
from .dimensions import Dimension, Dimensions
from .utils import Frozen
def resize_dimension(self, dim, size):
    """Resize a dimension to a certain size.

        It will pad with the underlying HDF5 data sets' fill values (usually
        zero) where necessary.
        """
    self._dimensions[dim]._resize(size)