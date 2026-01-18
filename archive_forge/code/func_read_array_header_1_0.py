import numpy
import warnings
from numpy.lib.utils import safe_eval, drop_metadata
from numpy.compat import (
def read_array_header_1_0(fp, max_header_size=_MAX_HEADER_SIZE):
    """
    Read an array header from a filelike object using the 1.0 file format
    version.

    This will leave the file object located just after the header.

    Parameters
    ----------
    fp : filelike object
        A file object or something with a `.read()` method like a file.

    Returns
    -------
    shape : tuple of int
        The shape of the array.
    fortran_order : bool
        The array data will be written out directly if it is either
        C-contiguous or Fortran-contiguous. Otherwise, it will be made
        contiguous before writing it out.
    dtype : dtype
        The dtype of the file's data.
    max_header_size : int, optional
        Maximum allowed size of the header.  Large headers may not be safe
        to load securely and thus require explicitly passing a larger value.
        See :py:func:`ast.literal_eval()` for details.

    Raises
    ------
    ValueError
        If the data is invalid.

    """
    return _read_array_header(fp, version=(1, 0), max_header_size=max_header_size)