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
def require_group(self, name):
    """Return a group, creating it if it doesn't exist.

        TypeError is raised if something with that name already exists that
        isn't a group.
        """
    with phil:
        if name not in self:
            return self.create_group(name)
        grp = self[name]
        if not isinstance(grp, Group):
            raise TypeError('Incompatible object (%s) already exists' % grp.__class__.__name__)
        return grp