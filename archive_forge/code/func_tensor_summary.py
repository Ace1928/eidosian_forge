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
def tensor_summary(tensor: _atypes.TensorFuzzingAnnotation[TV_TensorSummary_T], description: str='', labels=[], display_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Outputs a `Summary` protocol buffer with a tensor.

  This op is being phased out in favor of TensorSummaryV2, which lets callers pass
  a tag as well as a serialized SummaryMetadata proto string that contains
  plugin-specific data. We will keep this op to maintain backwards compatibility.

  Args:
    tensor: A `Tensor`. A tensor to serialize.
    description: An optional `string`. Defaults to `""`.
      A json-encoded SummaryDescription proto.
    labels: An optional list of `strings`. Defaults to `[]`.
      An unused list of strings.
    display_name: An optional `string`. Defaults to `""`. An unused string.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorSummary', name, tensor, 'description', description, 'labels', labels, 'display_name', display_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_summary_eager_fallback(tensor, description=description, labels=labels, display_name=display_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if description is None:
        description = ''
    description = _execute.make_str(description, 'description')
    if labels is None:
        labels = []
    if not isinstance(labels, (list, tuple)):
        raise TypeError("Expected list for 'labels' argument to 'tensor_summary' Op, not %r." % labels)
    labels = [_execute.make_str(_s, 'labels') for _s in labels]
    if display_name is None:
        display_name = ''
    display_name = _execute.make_str(display_name, 'display_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorSummary', tensor=tensor, description=description, labels=labels, display_name=display_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'description', _op.get_attr('description'), 'labels', _op.get_attr('labels'), 'display_name', _op.get_attr('display_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorSummary', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result