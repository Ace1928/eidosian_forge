import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def dtype_to_descr(dtype):
    """
    Get a serializable descriptor from the dtype.

    The .descr attribute of a dtype object cannot be round-tripped through
    the dtype() constructor. Simple types, like dtype('float32'), have
    a descr which looks like a record array with one field with '' as
    a name. The dtype() constructor interprets this as a request to give
    a default name.  Instead, we construct descriptor that can be passed to
    dtype().

    Parameters
    ----------
    dtype : dtype
        The dtype of the array that will be written to disk.

    Returns
    -------
    descr : object
        An object that can be passed to `numpy.dtype()` in order to
        replicate the input dtype.

    """
    new_dtype = drop_metadata(dtype)
    if new_dtype is not dtype:
        warnings.warn('metadata on a dtype is not saved to an npy/npz. Use another format (such as pickle) to store it.', UserWarning, stacklevel=2)
    if dtype.names is not None:
        return dtype.descr
    else:
        return dtype.str