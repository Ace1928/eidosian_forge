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
def adjust_contrast(images: _atypes.TensorFuzzingAnnotation[TV_AdjustContrast_T], contrast_factor: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min_value: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max_value: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Deprecated. Disallowed in GraphDef version >= 2.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
    contrast_factor: A `Tensor` of type `float32`.
    min_value: A `Tensor` of type `float32`.
    max_value: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AdjustContrast', name, images, contrast_factor, min_value, max_value)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return adjust_contrast_eager_fallback(images, contrast_factor, min_value, max_value, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AdjustContrast', images=images, contrast_factor=contrast_factor, min_value=min_value, max_value=max_value, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('AdjustContrast', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result