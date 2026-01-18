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
def sparse_softmax_cross_entropy_with_logits(features: _atypes.TensorFuzzingAnnotation[TV_SparseSoftmaxCrossEntropyWithLogits_T], labels: _atypes.TensorFuzzingAnnotation[TV_SparseSoftmaxCrossEntropyWithLogits_Tlabels], name=None):
    """Computes softmax cross entropy cost and gradients to backpropagate.

  Unlike `SoftmaxCrossEntropyWithLogits`, this operation does not accept
  a matrix of label probabilities, but rather a single label per row
  of features.  This label is considered to have probability 1.0 for the
  given row.

  Inputs are the logits, not probabilities.

  Args:
    features: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      batch_size x num_classes matrix
    labels: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      batch_size vector with values in [0, num_classes).
      This is the label for the given minibatch entry.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (loss, backprop).

    loss: A `Tensor`. Has the same type as `features`.
    backprop: A `Tensor`. Has the same type as `features`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SparseSoftmaxCrossEntropyWithLogits', name, features, labels)
            _result = _SparseSoftmaxCrossEntropyWithLogitsOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sparse_softmax_cross_entropy_with_logits_eager_fallback(features, labels, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SparseSoftmaxCrossEntropyWithLogits', features=features, labels=labels, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tlabels', _op._get_attr_type('Tlabels'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SparseSoftmaxCrossEntropyWithLogits', _inputs_flat, _attrs, _result)
    _result = _SparseSoftmaxCrossEntropyWithLogitsOutput._make(_result)
    return _result