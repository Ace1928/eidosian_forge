import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
def read_sub_array(self, hdr, copy=True):
    """ Mat4 read using header `hdr` dtype and dims

        Parameters
        ----------
        hdr : object
           object with attributes ``dtype``, ``dims``. dtype is assumed to be
           the correct endianness
        copy : bool, optional
           copies array before return if True (default True)
           (buffer is usually read only)

        Returns
        -------
        arr : ndarray
            of dtype given by `hdr` ``dtype`` and shape given by `hdr` ``dims``
        """
    dt = hdr.dtype
    dims = hdr.dims
    num_bytes = dt.itemsize
    for d in dims:
        num_bytes *= d
    buffer = self.mat_stream.read(int(num_bytes))
    if len(buffer) != num_bytes:
        raise ValueError("Not enough bytes to read matrix '%s'; is this a badly-formed file? Consider listing matrices with `whosmat` and loading named matrices with `variable_names` kwarg to `loadmat`" % hdr.name)
    arr = np.ndarray(shape=dims, dtype=dt, buffer=buffer, order='F')
    if copy:
        arr = arr.copy()
    return arr