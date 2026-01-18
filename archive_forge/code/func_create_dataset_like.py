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
def create_dataset_like(self, name, other, **kwupdate):
    """ Create a dataset similar to `other`.

        name
            Name of the dataset (absolute or relative).  Provide None to make
            an anonymous dataset.
        other
            The dataset which the new dataset should mimic. All properties, such
            as shape, dtype, chunking, ... will be taken from it, but no data
            or attributes are being copied.

        Any dataset keywords (see create_dataset) may be provided, including
        shape and dtype, in which case the provided values take precedence over
        those from `other`.
        """
    for k in ('shape', 'dtype', 'chunks', 'compression', 'compression_opts', 'scaleoffset', 'shuffle', 'fletcher32', 'fillvalue'):
        kwupdate.setdefault(k, getattr(other, k))
    dcpl = other.id.get_create_plist()
    kwupdate.setdefault('track_times', dcpl.get_obj_track_times())
    kwupdate.setdefault('track_order', dcpl.get_attr_creation_order() > 0)
    if other.maxshape != other.shape:
        kwupdate.setdefault('maxshape', other.maxshape)
    return self.create_dataset(name, **kwupdate)