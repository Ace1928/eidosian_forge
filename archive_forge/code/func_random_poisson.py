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
def random_poisson(shape: _atypes.TensorFuzzingAnnotation[TV_RandomPoisson_S], rate: _atypes.TensorFuzzingAnnotation[TV_RandomPoisson_dtype], seed: int=0, seed2: int=0, name=None) -> _atypes.TensorFuzzingAnnotation[TV_RandomPoisson_dtype]:
    """Use RandomPoissonV2 instead.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    rate: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `rate`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RandomPoisson', name, shape, rate, 'seed', seed, 'seed2', seed2)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return random_poisson_eager_fallback(shape, rate, seed=seed, seed2=seed2, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RandomPoisson', shape=shape, rate=rate, seed=seed, seed2=seed2, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'), 'S', _op._get_attr_type('S'), 'dtype', _op._get_attr_type('dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RandomPoisson', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result