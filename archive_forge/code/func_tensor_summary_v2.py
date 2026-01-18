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
def tensor_summary_v2(tag: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor: _atypes.TensorFuzzingAnnotation[TV_TensorSummaryV2_T], serialized_summary_metadata: _atypes.TensorFuzzingAnnotation[_atypes.String], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Outputs a `Summary` protocol buffer with a tensor and per-plugin data.

  Args:
    tag: A `Tensor` of type `string`.
      A string attached to this summary. Used for organization in TensorBoard.
    tensor: A `Tensor`. A tensor to serialize.
    serialized_summary_metadata: A `Tensor` of type `string`.
      A serialized SummaryMetadata proto. Contains plugin
      data.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorSummaryV2', name, tag, tensor, serialized_summary_metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_summary_v2_eager_fallback(tag, tensor, serialized_summary_metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorSummaryV2', tag=tag, tensor=tensor, serialized_summary_metadata=serialized_summary_metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorSummaryV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result