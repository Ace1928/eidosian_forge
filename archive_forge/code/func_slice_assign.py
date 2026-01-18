from array import array as native_array
import ctypes
import warnings
import operator
from functools import reduce # pylint: disable=redefined-builtin
import numpy as np
from ..base import _LIB, numeric_types, integer_types
from ..base import c_str, c_array, c_array_buf, c_handle_array, mx_real_t
from ..base import mx_uint, NDArrayHandle, check_call, DLPackHandle, mx_int, mx_int64
from ..base import ctypes2buffer
from ..runtime import Features
from ..context import Context, current_context
from ..util import is_np_array
from . import _internal
from . import op
from ._internal import NDArrayBase
def slice_assign(self, rhs, begin, end, step):
    """
        Assign the rhs to a cropped subset of this NDarray in place.
        Returns the view of this NDArray.

        Parameters
        ----------
        rhs: NDArray.
            rhs and this NDArray should be of the same data type, and on the same device.
            The shape of rhs should be the same as the cropped shape of this NDArray.
        begin: tuple of begin indices
        end: tuple of end indices
        step: tuple of step lenghths

        Returns
        -------
        This NDArray.

        Examples
        --------
        >>> x = nd.ones((2, 2, 2))
        >>> assigned = nd.zeros((1, 1, 2))
        >>> y = x.slice_assign(assigned, (0, 0, None), (1, 1, None), (None, None, None))
        >>> y
        [[[0. 0.]
        [1. 1.]]

        [[1. 1.]
        [1. 1.]]]
        <NDArray 2x2x2 @cpu(0)>
        >>> x
        [[[0. 0.]
        [1. 1.]]

        [[1. 1.]
        [1. 1.]]]
        <NDArray 2x2x2 @cpu(0)>
        """
    return _internal._slice_assign(self, rhs, begin=begin, end=end, step=step, out=self)