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
def ragged_tensor_to_variant(rt_nested_splits: List[_atypes.TensorFuzzingAnnotation[TV_RaggedTensorToVariant_Tsplits]], rt_dense_values: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToVariant_Tvalues], batched_input: bool, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Encodes a `RaggedTensor` into a `variant` Tensor.

  
  Encodes the given `RaggedTensor` and returns a `variant` Tensor. If
  `batched_input` is True, then input `RaggedTensor` is unbatched along the
  zero-th dimension, each component `RaggedTensor` is encoded into a scalar
  `variant` Tensor, and these are stacked to return a 1-D `variant` Tensor.
  If `batched_input` is False, then the input `RaggedTensor` is encoded as is and
  a scalar `variant` Tensor is returned. A `RaggedTensor` is encoded by first
  creating a 1-D `variant` Tensor with `ragged_rank + 1` elements, containing the
  splits and values Tensors of the `RaggedTensor`. Then the 1-D `variant` Tensor
  is wrapped in a scalar `variant` Tensor. See `RaggedTensorFromVariant` for the
  corresponding decoding logic.

  Args:
    rt_nested_splits: A list of `Tensor` objects with the same type in: `int32`, `int64`.
      A list of one or more Tensors representing the splits of the input
      `RaggedTensor`.
    rt_dense_values: A `Tensor`.
      A Tensor representing the values of the input `RaggedTensor`.
    batched_input: A `bool`.
      A `bool` denoting whether the input is a batched `RaggedTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedTensorToVariant', name, rt_nested_splits, rt_dense_values, 'batched_input', batched_input)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ragged_tensor_to_variant_eager_fallback(rt_nested_splits, rt_dense_values, batched_input=batched_input, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(rt_nested_splits, (list, tuple)):
        raise TypeError("Expected list for 'rt_nested_splits' argument to 'ragged_tensor_to_variant' Op, not %r." % rt_nested_splits)
    _attr_RAGGED_RANK = len(rt_nested_splits)
    batched_input = _execute.make_bool(batched_input, 'batched_input')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedTensorToVariant', rt_nested_splits=rt_nested_splits, rt_dense_values=rt_dense_values, batched_input=batched_input, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('RAGGED_RANK', _op._get_attr_int('RAGGED_RANK'), 'Tvalues', _op._get_attr_type('Tvalues'), 'Tsplits', _op._get_attr_type('Tsplits'), 'batched_input', _op._get_attr_bool('batched_input'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedTensorToVariant', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result