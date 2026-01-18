import itertools
import warnings
from . import core as ma
from .core import (
import numpy as np
from numpy import ndarray, array as nxarray
from numpy.core.multiarray import normalize_axis_index
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import _ureduce
from numpy.lib.index_tricks import AxisConcatenator
def masked_all(shape, dtype=float):
    """
    Empty masked array with all elements masked.

    Return an empty masked array of the given shape and dtype, where all the
    data are masked.

    Parameters
    ----------
    shape : int or tuple of ints
        Shape of the required MaskedArray, e.g., ``(2, 3)`` or ``2``.
    dtype : dtype, optional
        Data type of the output.

    Returns
    -------
    a : MaskedArray
        A masked array with all data masked.

    See Also
    --------
    masked_all_like : Empty masked array modelled on an existing array.

    Examples
    --------
    >>> import numpy.ma as ma
    >>> ma.masked_all((3, 3))
    masked_array(
      data=[[--, --, --],
            [--, --, --],
            [--, --, --]],
      mask=[[ True,  True,  True],
            [ True,  True,  True],
            [ True,  True,  True]],
      fill_value=1e+20,
      dtype=float64)

    The `dtype` parameter defines the underlying data type.

    >>> a = ma.masked_all((3, 3))
    >>> a.dtype
    dtype('float64')
    >>> a = ma.masked_all((3, 3), dtype=np.int32)
    >>> a.dtype
    dtype('int32')

    """
    a = masked_array(np.empty(shape, dtype), mask=np.ones(shape, make_mask_descr(dtype)))
    return a