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
def kmc2_chain_initialization(distances: _atypes.TensorFuzzingAnnotation[_atypes.Float32], seed: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    """Returns the index of a data point that should be added to the seed set.

  Entries in distances are assumed to be squared distances of candidate points to
  the already sampled centers in the seed set. The op constructs one Markov chain
  of the k-MC^2 algorithm and returns the index of one candidate point to be added
  as an additional cluster center.

  Args:
    distances: A `Tensor` of type `float32`.
      Vector with squared distances to the closest previously sampled cluster center
      for each candidate point.
    seed: A `Tensor` of type `int64`.
      Scalar. Seed for initializing the random number generator.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'KMC2ChainInitialization', name, distances, seed)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return kmc2_chain_initialization_eager_fallback(distances, seed, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('KMC2ChainInitialization', distances=distances, seed=seed, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('KMC2ChainInitialization', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result