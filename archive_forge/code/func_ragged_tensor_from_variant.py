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
def ragged_tensor_from_variant(encoded_ragged: _atypes.TensorFuzzingAnnotation[_atypes.Variant], input_ragged_rank: int, output_ragged_rank: int, Tvalues: TV_RaggedTensorFromVariant_Tvalues, Tsplits: TV_RaggedTensorFromVariant_Tsplits=_dtypes.int64, name=None):
    """Decodes a `variant` Tensor into a `RaggedTensor`.

  Decodes the given `variant` Tensor and returns a `RaggedTensor`. The input
  could be a scalar, meaning it encodes a single `RaggedTensor` with ragged_rank
  `output_ragged_rank`. It could also have an arbitrary rank, in which case each
  element is decoded into a `RaggedTensor` with ragged_rank `input_ragged_rank`
  and these are then stacked according to the input shape to output a single
  `RaggedTensor` with ragged_rank `output_ragged_rank`. Each `variant` element in
  the input Tensor is decoded by retrieving from the element a 1-D `variant`
  Tensor with `input_ragged_rank + 1` Tensors, corresponding to the splits and
  values of the decoded `RaggedTensor`. If `input_ragged_rank` is -1, then it is
  inferred as `output_ragged_rank` - `rank(encoded_ragged)`. See
  `RaggedTensorToVariant` for the corresponding encoding logic.

  Args:
    encoded_ragged: A `Tensor` of type `variant`.
      A `variant` Tensor containing encoded `RaggedTensor`s.
    input_ragged_rank: An `int` that is `>= -1`.
      The ragged rank of each encoded `RaggedTensor` component in the input. If set to
      -1, this is inferred as `output_ragged_rank` - `rank(encoded_ragged)`
    output_ragged_rank: An `int` that is `>= 0`.
      The expected ragged rank of the output `RaggedTensor`. The following must hold:
      `output_ragged_rank = rank(encoded_ragged) + input_ragged_rank`.
    Tvalues: A `tf.DType`.
    Tsplits: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_nested_splits, output_dense_values).

    output_nested_splits: A list of `output_ragged_rank` `Tensor` objects with type `Tsplits`.
    output_dense_values: A `Tensor` of type `Tvalues`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedTensorFromVariant', name, encoded_ragged, 'input_ragged_rank', input_ragged_rank, 'output_ragged_rank', output_ragged_rank, 'Tvalues', Tvalues, 'Tsplits', Tsplits)
            _result = _RaggedTensorFromVariantOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ragged_tensor_from_variant_eager_fallback(encoded_ragged, input_ragged_rank=input_ragged_rank, output_ragged_rank=output_ragged_rank, Tvalues=Tvalues, Tsplits=Tsplits, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    input_ragged_rank = _execute.make_int(input_ragged_rank, 'input_ragged_rank')
    output_ragged_rank = _execute.make_int(output_ragged_rank, 'output_ragged_rank')
    Tvalues = _execute.make_type(Tvalues, 'Tvalues')
    if Tsplits is None:
        Tsplits = _dtypes.int64
    Tsplits = _execute.make_type(Tsplits, 'Tsplits')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedTensorFromVariant', encoded_ragged=encoded_ragged, input_ragged_rank=input_ragged_rank, output_ragged_rank=output_ragged_rank, Tvalues=Tvalues, Tsplits=Tsplits, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('input_ragged_rank', _op._get_attr_int('input_ragged_rank'), 'output_ragged_rank', _op._get_attr_int('output_ragged_rank'), 'Tvalues', _op._get_attr_type('Tvalues'), 'Tsplits', _op._get_attr_type('Tsplits'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedTensorFromVariant', _inputs_flat, _attrs, _result)
    _result = [_result[:output_ragged_rank]] + _result[output_ragged_rank:]
    _result = _RaggedTensorFromVariantOutput._make(_result)
    return _result