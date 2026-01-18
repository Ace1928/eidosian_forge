from contextlib import contextmanager
import posixpath as pp
import numpy
from .compat import filename_decode, filename_encode
from .. import h5, h5g, h5i, h5o, h5r, h5t, h5l, h5p
from . import base
from .base import HLObject, MutableMappingHDF5, phil, with_phil
from . import dataset
from . import datatype
from .vds import vds_support
def create_group(self, name, track_order=None):
    """ Create and return a new subgroup.

        Name may be absolute or relative.  Fails if the target name already
        exists.

        track_order
            Track dataset/group/attribute creation order under this group
            if True. If None use global default h5.get_config().track_order.
        """
    if track_order is None:
        track_order = h5.get_config().track_order
    with phil:
        name, lcpl = self._e(name, lcpl=True)
        gcpl = Group._gcpl_crt_order if track_order else None
        gid = h5g.create(self.id, name, lcpl=lcpl, gcpl=gcpl)
        return Group(gid)