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
def attach_grad(self, grad_req='write', stype=None):
    """Attach a gradient buffer to this NDArray, so that `backward`
        can compute gradient with respect to it.

        The gradient is initialized to zeros.

        Parameters
        ----------
        grad_req : {'write', 'add', 'null'}
            How gradient will be accumulated.
            - 'write': gradient will be overwritten on every backward.
            - 'add': gradient will be added to existing value on every backward.
            - 'null': do not compute gradient for this NDArray.
        stype : str, optional
            The storage type of the gradient array. Defaults to the same stype of this NDArray.
        """
    from . import zeros as _zeros
    if stype is not None:
        grad = _zeros(self.shape, stype=stype)
    else:
        grad = op.zeros_like(self)
    grad_req = _GRAD_REQ_MAP[grad_req]
    check_call(_LIB.MXAutogradMarkVariables(1, ctypes.pointer(self.handle), ctypes.pointer(mx_uint(grad_req)), ctypes.pointer(grad.handle)))