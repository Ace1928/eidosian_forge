import ctypes
import numpy as _np
from . import _op as _mx_np_op
from ...base import _LIB, SymbolHandle, numeric_types, mx_uint, integer_types, string_types
from ...base import c_str
from ...base import py_str
from ...util import check_call, set_module, _sanity_check_params
from ...util import wrap_np_unary_func, wrap_np_binary_func
from ...context import current_context
from ..symbol import Symbol, Group
from .._internal import _set_np_symbol_class
from . import _internal as _npi
def shape_array(self, *args, **kwargs):
    """Convenience fluent method for :py:func:`shape_array`.

        The arguments are the same as for :py:func:`shape_array`, with
        this array as data.
        """
    raise AttributeError('_Symbol object has no attribute shape_array')