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
def recv_tpu_embedding_activations(num_outputs: int, config: str, name=None):
    """An op that receives embedding activations on the TPU.

  The TPU system performs the embedding lookups and aggregations specified by
  the arguments to TPUEmbeddingEnqueue(Integer/Sparse/SparseTensor)Batch. The
  results of these aggregations are visible to the Tensorflow Graph as the
  outputs of a RecvTPUEmbeddingActivations op. This op returns a list containing
  one Tensor of activations per table specified in the model. There can be at
  most one RecvTPUEmbeddingActivations op in the TPU graph.

  Args:
    num_outputs: An `int` that is `>= 1`.
      The number of output activation tensors, equal to the number of
      embedding tables in the model.
    config: A `string`. Serialized TPUEmbeddingConfiguration proto.
    name: A name for the operation (optional).

  Returns:
    A list of `num_outputs` `Tensor` objects with type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RecvTPUEmbeddingActivations', name, 'num_outputs', num_outputs, 'config', config)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return recv_tpu_embedding_activations_eager_fallback(num_outputs=num_outputs, config=config, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_outputs = _execute.make_int(num_outputs, 'num_outputs')
    config = _execute.make_str(config, 'config')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RecvTPUEmbeddingActivations', num_outputs=num_outputs, config=config, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('num_outputs', _op._get_attr_int('num_outputs'), 'config', _op.get_attr('config'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RecvTPUEmbeddingActivations', _inputs_flat, _attrs, _result)
    return _result