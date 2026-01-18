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
def infer_shape_partial(self, *args, **kwargs):
    """Infers the shape partially.

        This functions works the same way as `infer_shape`,
        except that this function can return partial results.

        In the following example, information about fc2 is not available. So, `infer_shape`
        will return a tuple of `None` values but `infer_shape_partial` will return partial values.

        Example
        -------
        >>> data = mx.sym.Variable('data')
        >>> prev = mx.sym.Variable('prev')
        >>> fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
        >>> fc2  = mx.sym.FullyConnected(data=prev, name='fc2', num_hidden=128)
        >>> out  = mx.sym.Activation(data=mx.sym.elemwise_add(fc1, fc2), act_type='relu')
        >>> out.list_arguments()
        ['data', 'fc1_weight', 'fc1_bias', 'prev', 'fc2_weight', 'fc2_bias']
        >>> out.infer_shape(data=(10,64))
        (None, None, None)
        >>> out.infer_shape_partial(data=(10,64))
        ([(10L, 64L), (128L, 64L), (128L,), (), (), ()], [(10L, 128L)], [])
        >>> # infers shape if you give information about fc2
        >>> out.infer_shape(data=(10,64), prev=(10,128))
        ([(10L, 64L), (128L, 64L), (128L,), (10L, 128L), (128L, 128L), (128L,)], [(10L, 128L)], [])

        Parameters
        ----------
        *args :
            Shape of arguments in a positional way.
            Unknown shape can be marked as None

        **kwargs :
            Keyword arguments of known shapes.

        Returns
        -------
        arg_shapes : list of tuple or None
            List of argument shapes.
            The order is same as the order of list_arguments().
        out_shapes : list of tuple or None
            List of output shapes.
            The order is same as the order of list_outputs().
        aux_shapes : list of tuple or None
            List of auxiliary state shapes.
            The order is same as the order of list_auxiliary_states().
        """
    return self._infer_shape_impl(True, *args, **kwargs)