import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def concat_offset(concat_dim: _atypes.TensorFuzzingAnnotation[_atypes.Int32], shape: List[_atypes.TensorFuzzingAnnotation[TV_ConcatOffset_shape_type]], name=None):
    """Computes offsets of concat inputs within its output.

  For example:

  >>> x = [2, 2, 7]
  >>> y = [2, 3, 7]
  >>> z = [2, 9, 7]
  >>> offsets = concat_offset(1, [x, y, z])
  >>> [list(off.numpy()) for off in offsets]
  [[0, 0, 0], [0, 2, 0], [0, 5, 0]]

  This is typically used by gradient computations for a concat operation.

  Args:
    concat_dim: A `Tensor` of type `int32`.
      The dimension along which to concatenate.
    shape: A list of at least 2 `Tensor` objects with the same type in: `int32`, `int64`.
      The `N` int32 or int64 vectors representing shape of tensors being concatenated.
    name: A name for the operation (optional).

  Returns:
    A list with the same length as `shape` of `Tensor` objects with the same type as `shape`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ConcatOffset', name, concat_dim, shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return concat_offset_eager_fallback(concat_dim, shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(shape, (list, tuple)):
        raise TypeError("Expected list for 'shape' argument to 'concat_offset' Op, not %r." % shape)
    _attr_N = len(shape)
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ConcatOffset', concat_dim=concat_dim, shape=shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'), 'shape_type', _op._get_attr_type('shape_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ConcatOffset', _inputs_flat, _attrs, _result)
    return _result