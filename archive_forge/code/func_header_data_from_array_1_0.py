import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def header_data_from_array_1_0(array):
    """ Get the dictionary of header metadata from a numpy.ndarray.

    Parameters
    ----------
    array : numpy.ndarray

    Returns
    -------
    d : dict
        This has the appropriate entries for writing its string representation
        to the header of the file.
    """
    d = {'shape': array.shape}
    if array.flags.c_contiguous:
        d['fortran_order'] = False
    elif array.flags.f_contiguous:
        d['fortran_order'] = True
    else:
        d['fortran_order'] = False
    d['descr'] = dtype_to_descr(array.dtype)
    return d