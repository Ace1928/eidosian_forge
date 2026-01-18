import posixpath as pp
import sys
import numpy
from .. import h5, h5s, h5t, h5r, h5d, h5p, h5fd, h5ds, _selector
from .base import (
from . import filters
from . import selections as sel
from . import selections2 as sel2
from .datatype import Datatype
from .compat import filename_decode
from .vds import VDSmap, vds_support
@property
@with_phil
def fillvalue(self):
    """Fill value for this dataset (0 by default)"""
    arr = numpy.zeros((1,), dtype=self.dtype)
    self._dcpl.get_fill_value(arr)
    return arr[0]