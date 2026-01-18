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
def maxshape(self):
    """Shape up to which this dataset can be resized.  Axes with value
        None have no resize limit. """
    space = self.id.get_space()
    dims = space.get_simple_extent_dims(True)
    if dims is None:
        return None
    return tuple((x if x != h5s.UNLIMITED else None for x in dims))