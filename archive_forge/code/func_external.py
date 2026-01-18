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
def external(self):
    """External file settings. Returns a list of tuples of
        (name, offset, size) for each external file entry, or returns None
        if no external files are used."""
    count = self._dcpl.get_external_count()
    if count <= 0:
        return None
    ext_list = list()
    for x in range(count):
        name, offset, size = self._dcpl.get_external(x)
        ext_list.append((filename_decode(name), offset, size))
    return ext_list