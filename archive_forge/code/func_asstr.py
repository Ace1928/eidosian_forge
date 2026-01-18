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
def asstr(self, encoding=None, errors='strict'):
    """Get a wrapper to read string data as Python strings:

        >>> str_array = dataset.asstr()[:]

        The parameters have the same meaning as in ``bytes.decode()``.
        If ``encoding`` is unspecified, it will use the encoding in the HDF5
        datatype (either ascii or utf-8).
        """
    string_info = h5t.check_string_dtype(self.dtype)
    if string_info is None:
        raise TypeError('dset.asstr() can only be used on datasets with an HDF5 string datatype')
    if encoding is None:
        encoding = string_info.encoding
    return AsStrWrapper(self, encoding, errors=errors)