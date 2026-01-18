import warnings
from .. import h5ds
from ..h5py_warnings import H5pyDeprecationWarning
from . import base
from .base import phil, with_phil
from .dataset import Dataset
def attach_scale(self, dset):
    """ Attach a scale to this dimension.

        Provide the Dataset of the scale you would like to attach.
        """
    with phil:
        h5ds.attach_scale(self._id, dset.id, self._dimension)