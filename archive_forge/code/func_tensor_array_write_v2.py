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
def tensor_array_write_v2(handle: _atypes.TensorFuzzingAnnotation[_atypes.String], index: _atypes.TensorFuzzingAnnotation[_atypes.Int32], value: _atypes.TensorFuzzingAnnotation[TV_TensorArrayWriteV2_T], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Deprecated. Use TensorArrayGradV3

  Args:
    handle: A `Tensor` of type `string`.
    index: A `Tensor` of type `int32`.
    value: A `Tensor`.
    flow_in: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorArrayWriteV2', name, handle, index, value, flow_in)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_array_write_v2_eager_fallback(handle, index, value, flow_in, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArrayWriteV2', handle=handle, index=index, value=value, flow_in=flow_in, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArrayWriteV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result