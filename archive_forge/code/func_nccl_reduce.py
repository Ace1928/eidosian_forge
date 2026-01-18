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
def nccl_reduce(input: List[_atypes.TensorFuzzingAnnotation[TV_NcclReduce_T]], reduction: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_NcclReduce_T]:
    """Reduces `input` from `num_devices` using `reduction` to a single device.

  Reduces `input` from `num_devices` using `reduction` to a single device.

  The graph should be constructed so that all inputs have a valid device
  assignment, and the op itself is assigned one of these devices.

  input: The input to the reduction.
  data: the value of the reduction across all `num_devices` devices.
  reduction: the reduction operation to perform.

  Args:
    input: A list of at least 1 `Tensor` objects with the same type in: `half`, `float32`, `float64`, `int32`, `int64`.
    reduction: A `string` from: `"min", "max", "prod", "sum"`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'NcclReduce', name, input, 'reduction', reduction)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return nccl_reduce_eager_fallback(input, reduction=reduction, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(input, (list, tuple)):
        raise TypeError("Expected list for 'input' argument to 'nccl_reduce' Op, not %r." % input)
    _attr_num_devices = len(input)
    reduction = _execute.make_str(reduction, 'reduction')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('NcclReduce', input=input, reduction=reduction, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('reduction', _op.get_attr('reduction'), 'T', _op._get_attr_type('T'), 'num_devices', _op._get_attr_int('num_devices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('NcclReduce', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result