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
def attr_dict(self):
    """Recursively gets all attributes from the symbol and its children.

        Example
        -------
        >>> a = mx.sym.Variable('a', attr={'a1':'a2'})
        >>> b = mx.sym.Variable('b', attr={'b1':'b2'})
        >>> c = a+b
        >>> c.attr_dict()
        {'a': {'a1': 'a2'}, 'b': {'b1': 'b2'}}

        Returns
        -------
        ret : Dict of str to dict
            There is a key in the returned dict for every child with non-empty attribute set.
            For each symbol, the name of the symbol is its key in the dict
            and the correspond value is that symbol's attribute list (itself a dictionary).
        """
    size = mx_uint()
    pairs = ctypes.POINTER(ctypes.c_char_p)()
    f_handle = _LIB.MXSymbolListAttr
    check_call(f_handle(self.handle, ctypes.byref(size), ctypes.byref(pairs)))
    ret = {}
    for i in range(size.value):
        name, key = py_str(pairs[i * 2]).split('$')
        val = py_str(pairs[i * 2 + 1])
        if name not in ret:
            ret[name] = {}
        ret[name][key] = val
    return ret