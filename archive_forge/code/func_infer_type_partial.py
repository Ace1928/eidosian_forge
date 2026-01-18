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
def infer_type_partial(self, *args, **kwargs):
    """Infers the type partially.

        This functions works the same way as `infer_type`,
        except that this function can return partial results.

        In the following example, information about fc2 is not available. So, `infer_shape`
        will return a tuple of `None` values but `infer_shape_partial` will return partial values.

        Example
        -------
        >>> data = mx.sym.Variable('data')
        >>> prev = mx.sym.Variable('prev')
        >>> casted_prev  = mx.sym.cast(prev, dtype='float32')
        >>> out  = mx.sym.Activation(data=mx.sym.elemwise_add(data, casted_prev), act_type='relu')
        >>> out.list_arguments()
        ['data', 'prev']
        >>> out.infer_type(data='float32')
        (None, None, None)
        >>> out.infer_type_partial(data='float32')
        ([numpy.float32, None], [numpy.float32], [])
        >>> # infers type if you give information about prev
        >>> out.infer_type(data='float32', prev='float16')
        ([numpy.float32, numpy.float16], [numpy.float32], [])

        Parameters
        ----------
        *args :
            Type of known arguments in a positional way.
            Unknown type can be marked as None.

        **kwargs :
            Keyword arguments of known types.

        Returns
        -------
        arg_types : list of numpy.dtype or None
            List of argument types.
            The order is same as the order of list_arguments().
        out_types : list of numpy.dtype or None
            List of output types.
            The order is same as the order of list_outputs().
        aux_types : list of numpy.dtype or None
            List of auxiliary state types.
            The order is same as the order of list_auxiliary_states().
        """
    return self._infer_type_impl(True, *args, **kwargs)