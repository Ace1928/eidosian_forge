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
def ragged_tensor_to_variant_gradient(encoded_ragged_grad: _atypes.TensorFuzzingAnnotation[_atypes.Variant], row_splits: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToVariantGradient_Tsplits], dense_values_shape: _atypes.TensorFuzzingAnnotation[_atypes.Int32], Tvalues: TV_RaggedTensorToVariantGradient_Tvalues, name=None) -> _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToVariantGradient_Tvalues]:
    """Helper used to compute the gradient for `RaggedTensorToVariant`.

  Computes the gradient for the dense_values input to the RaggedTensorToVariant
  op, given the variant-encoded ragged gradients of the outputs, along with
  the outer row-splits and the shape of the dense-values that were provided as
  inputs to the RaggedTensorToVariant op.

  Args:
    encoded_ragged_grad: A `Tensor` of type `variant`.
      A `variant` Tensor containing encoded `RaggedTensor` gradients.
    row_splits: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Outermost row-splits that were used as input to the RaggedTensorToVariant op.
    dense_values_shape: A `Tensor` of type `int32`.
      Shape of the dense_values that was used as an input to the
      RaggedTensorToVariant op.
    Tvalues: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `Tvalues`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedTensorToVariantGradient', name, encoded_ragged_grad, row_splits, dense_values_shape, 'Tvalues', Tvalues)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ragged_tensor_to_variant_gradient_eager_fallback(encoded_ragged_grad, row_splits, dense_values_shape, Tvalues=Tvalues, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    Tvalues = _execute.make_type(Tvalues, 'Tvalues')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedTensorToVariantGradient', encoded_ragged_grad=encoded_ragged_grad, row_splits=row_splits, dense_values_shape=dense_values_shape, Tvalues=Tvalues, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tvalues', _op._get_attr_type('Tvalues'), 'Tsplits', _op._get_attr_type('Tsplits'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedTensorToVariantGradient', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result