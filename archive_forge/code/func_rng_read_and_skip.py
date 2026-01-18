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
def rng_read_and_skip(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], alg: _atypes.TensorFuzzingAnnotation[_atypes.Int32], delta: _atypes.TensorFuzzingAnnotation[_atypes.UInt64], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    """Advance the counter of a counter-based RNG.

  The state of the RNG after
  `rng_read_and_skip(n)` will be the same as that after `uniform([n])`
  (or any other distribution). The actual increment added to the
  counter is an unspecified implementation choice.

  In the case that the input algorithm is RNG_ALG_AUTO_SELECT, the counter in the state needs to be of size int64[2], the current maximal counter size among algorithms. In this case, this op will manage the counter as if it is an 128-bit integer with layout [lower_64bits, higher_64bits]. If an algorithm needs less than 128 bits for the counter, it should use the left portion of the int64[2]. In this way, the int64[2] is compatible with all current RNG algorithms (Philox, ThreeFry and xla::RandomAlgorithm::RNG_DEFAULT). Downstream RNG ops can thus use this counter with any RNG algorithm.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG. The state consists of the counter followed by the key.
    alg: A `Tensor` of type `int32`. The RNG algorithm.
    delta: A `Tensor` of type `uint64`. The amount of advancement.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RngReadAndSkip', name, resource, alg, delta)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return rng_read_and_skip_eager_fallback(resource, alg, delta, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RngReadAndSkip', resource=resource, alg=alg, delta=delta, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('RngReadAndSkip', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result