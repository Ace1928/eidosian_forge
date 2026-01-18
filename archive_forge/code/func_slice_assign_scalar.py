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
def slice_assign_scalar(self, value, begin, end, step):
    """
        Assign the scalar to a cropped subset of this NDArray. Value will broadcast to the shape of the cropped shape
        and will be cast to the same dtype of the NDArray.

        Parameters
        ----------
        value: numeric value
            Value and this NDArray should be of the same data type.
            The shape of rhs should be the same as the cropped shape of this NDArray.
        begin: tuple of begin indices
        end: tuple of end indices
        step: tuple of step lenghths

        Returns
        -------
        This NDArray.

        Examples
        --------
        >>> from mxnet import nd
        >>> x = nd.ones((2, 2, 2))
        >>> y = x.slice_assign_scalar(0, (0, 0, None), (1, 1, None), (None, None, None))
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
    return _internal._slice_assign_scalar(self, value, begin=begin, end=end, step=step, out=self)