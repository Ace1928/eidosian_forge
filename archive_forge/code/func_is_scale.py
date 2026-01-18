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
def is_scale(self):
    """Return ``True`` if this dataset is also a dimension scale.

        Return ``False`` otherwise.
        """
    return h5ds.is_scale(self._id)