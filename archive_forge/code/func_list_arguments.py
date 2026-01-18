from array import array
import ctypes
import warnings
from numbers import Number
import numpy as _numpy  # pylint: disable=relative-import
from ..attribute import AttrScope
from ..base import _LIB, numeric_types, c_array, c_array_buf, c_str, c_str_array, c_handle_array
from ..base import mx_uint, py_str, string_types, integer_types, mx_int, mx_int64
from ..base import NDArrayHandle, ExecutorHandle, SymbolHandle
from ..base import check_call, MXNetError, NotImplementedForSymbol
from ..context import Context, current_context
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, _GRAD_REQ_MAP
from ..ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _int64_enabled, _SIGNED_INT32_UPPER_LIMIT
from ..ndarray import _ndarray_cls
from ..executor import Executor
from . import _internal
from . import op
from ._internal import SymbolBase, _set_symbol_class
from ..util import is_np_shape
def list_arguments(self):
    """Lists all the arguments in the symbol.

        Example
        -------
        >>> a = mx.sym.var('a')
        >>> b = mx.sym.var('b')
        >>> c = a + b
        >>> c.list_arguments
        ['a', 'b']

        Returns
        -------
        args : list of string
            List containing the names of all the arguments required to compute the symbol.
        """
    size = ctypes.c_uint()
    sarr = ctypes.POINTER(ctypes.c_char_p)()
    check_call(_LIB.MXSymbolListArguments(self.handle, ctypes.byref(size), ctypes.byref(sarr)))
    return [py_str(sarr[i]) for i in range(size.value)]