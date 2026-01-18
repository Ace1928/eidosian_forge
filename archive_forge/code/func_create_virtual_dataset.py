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
def create_virtual_dataset(self, name, layout, fillvalue=None):
    """Create a new virtual dataset in this group.

            See virtual datasets in the docs for more information.

            name
                (str) Name of the new dataset

            layout
                (VirtualLayout) Defines the sources for the virtual dataset

            fillvalue
                The value to use where there is no data.

            """
    with phil:
        group = self
        if name:
            name = self._e(name)
            if b'/' in name.lstrip(b'/'):
                parent_path, name = name.rsplit(b'/', 1)
                group = self.require_group(parent_path)
        dsid = layout.make_dataset(group, name=name, fillvalue=fillvalue)
        dset = dataset.Dataset(dsid)
    return dset