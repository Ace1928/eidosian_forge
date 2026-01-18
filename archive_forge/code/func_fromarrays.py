from numpy.ma import (
import numpy.ma as ma
import warnings
import numpy as np
from numpy import (
from numpy.core.records import (
def fromarrays(arraylist, dtype=None, shape=None, formats=None, names=None, titles=None, aligned=False, byteorder=None, fill_value=None):
    """
    Creates a mrecarray from a (flat) list of masked arrays.

    Parameters
    ----------
    arraylist : sequence
        A list of (masked) arrays. Each element of the sequence is first converted
        to a masked array if needed. If a 2D array is passed as argument, it is
        processed line by line
    dtype : {None, dtype}, optional
        Data type descriptor.
    shape : {None, integer}, optional
        Number of records. If None, shape is defined from the shape of the
        first array in the list.
    formats : {None, sequence}, optional
        Sequence of formats for each individual field. If None, the formats will
        be autodetected by inspecting the fields and selecting the highest dtype
        possible.
    names : {None, sequence}, optional
        Sequence of the names of each field.
    fill_value : {None, sequence}, optional
        Sequence of data to be used as filling values.

    Notes
    -----
    Lists of tuples should be preferred over lists of lists for faster processing.

    """
    datalist = [getdata(x) for x in arraylist]
    masklist = [np.atleast_1d(getmaskarray(x)) for x in arraylist]
    _array = recfromarrays(datalist, dtype=dtype, shape=shape, formats=formats, names=names, titles=titles, aligned=aligned, byteorder=byteorder).view(mrecarray)
    _array._mask.flat = list(zip(*masklist))
    if fill_value is not None:
        _array.fill_value = fill_value
    return _array