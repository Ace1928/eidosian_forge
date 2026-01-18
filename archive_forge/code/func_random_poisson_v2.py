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
def random_poisson_v2(shape: _atypes.TensorFuzzingAnnotation[TV_RandomPoissonV2_S], rate: _atypes.TensorFuzzingAnnotation[TV_RandomPoissonV2_R], seed: int=0, seed2: int=0, dtype: TV_RandomPoissonV2_dtype=_dtypes.int64, name=None) -> _atypes.TensorFuzzingAnnotation[TV_RandomPoissonV2_dtype]:
    """Outputs random values from the Poisson distribution(s) described by rate.

  This op uses two algorithms, depending on rate. If rate >= 10, then
  the algorithm by Hormann is used to acquire samples via
  transformation-rejection.
  See http://www.sciencedirect.com/science/article/pii/0167668793909974.

  Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
  random variables.
  See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
  Programming, Volume 2. Addison Wesley

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      1-D integer tensor. Shape of independent samples to draw from each
      distribution described by the shape parameters given in rate.
    rate: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
      A tensor in which each scalar is a "rate" parameter describing the
      associated poisson distribution.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    dtype: An optional `tf.DType` from: `tf.half, tf.float32, tf.float64, tf.int32, tf.int64`. Defaults to `tf.int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RandomPoissonV2', name, shape, rate, 'seed', seed, 'seed2', seed2, 'dtype', dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return random_poisson_v2_eager_fallback(shape, rate, seed=seed, seed2=seed2, dtype=dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    if dtype is None:
        dtype = _dtypes.int64
    dtype = _execute.make_type(dtype, 'dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RandomPoissonV2', shape=shape, rate=rate, seed=seed, seed2=seed2, dtype=dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'), 'S', _op._get_attr_type('S'), 'R', _op._get_attr_type('R'), 'dtype', _op._get_attr_type('dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RandomPoissonV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result