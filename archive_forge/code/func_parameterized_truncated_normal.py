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
def parameterized_truncated_normal(shape: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_T], means: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype], stdevs: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype], minvals: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype], maxvals: _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype], seed: int=0, seed2: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ParameterizedTruncatedNormal_dtype]:
    """Outputs random values from a normal distribution. The parameters may each be a

  scalar which applies to the entire output, or a vector of length shape[0] which
  stores the parameters for each batch.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor. Batches are indexed by the 0th dimension.
    means: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      The mean parameter of each batch.
    stdevs: A `Tensor`. Must have the same type as `means`.
      The standard deviation parameter of each batch. Must be greater than 0.
    minvals: A `Tensor`. Must have the same type as `means`.
      The minimum cutoff. May be -infinity.
    maxvals: A `Tensor`. Must have the same type as `means`.
      The maximum cutoff. May be +infinity, and must be more than the minval
      for each batch.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `means`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ParameterizedTruncatedNormal', name, shape, means, stdevs, minvals, maxvals, 'seed', seed, 'seed2', seed2)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return parameterized_truncated_normal_eager_fallback(shape, means, stdevs, minvals, maxvals, seed=seed, seed2=seed2, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ParameterizedTruncatedNormal', shape=shape, means=means, stdevs=stdevs, minvals=minvals, maxvals=maxvals, seed=seed, seed2=seed2, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'), 'dtype', _op._get_attr_type('dtype'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ParameterizedTruncatedNormal', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result