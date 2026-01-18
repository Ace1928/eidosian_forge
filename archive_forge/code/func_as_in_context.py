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
def as_in_context(self, context):
    """Returns an array on the target device with the same value as this array.

        If the target context is the same as ``self.context``, then ``self`` is
        returned.  Otherwise, a copy is made.

        Parameters
        ----------
        context : Context
            The target context.

        Returns
        -------
        NDArray, CSRNDArray or RowSparseNDArray
            The target array.


        Examples
        --------
        >>> x = mx.nd.ones((2,3))
        >>> y = x.as_in_context(mx.cpu())
        >>> y is x
        True
        >>> z = x.as_in_context(mx.gpu(0))
        >>> z is x
        False
        """
    if self.context == context:
        return self
    return self.copyto(context)