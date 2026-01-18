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
def stateless_random_gamma_v3(shape: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV3_shape_dtype], key: _atypes.TensorFuzzingAnnotation[_atypes.UInt64], counter: _atypes.TensorFuzzingAnnotation[_atypes.UInt64], alg: _atypes.TensorFuzzingAnnotation[_atypes.Int32], alpha: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV3_dtype], name=None) -> _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV3_dtype]:
    """Outputs deterministic pseudorandom random numbers from a gamma distribution.

  Outputs random values from a gamma distribution.

  The outputs are a deterministic function of the inputs.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    key: A `Tensor` of type `uint64`.
      Key for the counter-based RNG algorithm (shape uint64[1]).
    counter: A `Tensor` of type `uint64`.
      Initial counter for the counter-based RNG algorithm (shape uint64[2] or uint64[1] depending on the algorithm). If a larger vector is given, only the needed portion on the left (i.e. [:N]) will be used.
    alg: A `Tensor` of type `int32`. The RNG algorithm (shape int32[]).
    alpha: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      The concentration of the gamma distribution. Shape must match the rightmost
      dimensions of `shape`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `alpha`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatelessRandomGammaV3', name, shape, key, counter, alg, alpha)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateless_random_gamma_v3_eager_fallback(shape, key, counter, alg, alpha, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatelessRandomGammaV3', shape=shape, key=key, counter=counter, alg=alg, alpha=alpha, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'shape_dtype', _op._get_attr_type('shape_dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatelessRandomGammaV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result