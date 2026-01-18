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
def tensor_array_split_v3(handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], value: _atypes.TensorFuzzingAnnotation[TV_TensorArraySplitV3_T], lengths: _atypes.TensorFuzzingAnnotation[_atypes.Int64], flow_in: _atypes.TensorFuzzingAnnotation[_atypes.Float32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Split the data from the input value into TensorArray elements.

  Assuming that `lengths` takes on values

    ```
    (n0, n1, ..., n(T-1))
    ```

  and that `value` has shape

    ```
    (n0 + n1 + ... + n(T-1) x d0 x d1 x ...),
    ```

  this splits values into a TensorArray with T tensors.

  TensorArray index t will be the subtensor of values with starting position

    ```
    (n0 + n1 + ... + n(t-1), 0, 0, ...)
    ```

  and having size

    ```
    nt x d0 x d1 x ...
    ```

  Args:
    handle: A `Tensor` of type `resource`. The handle to a TensorArray.
    value: A `Tensor`. The concatenated tensor to write to the TensorArray.
    lengths: A `Tensor` of type `int64`.
      The vector of lengths, how to split the rows of value into the
      TensorArray.
    flow_in: A `Tensor` of type `float32`.
      A float scalar that enforces proper chaining of operations.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorArraySplitV3', name, handle, value, lengths, flow_in)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_array_split_v3_eager_fallback(handle, value, lengths, flow_in, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorArraySplitV3', handle=handle, value=value, lengths=lengths, flow_in=flow_in, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorArraySplitV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result