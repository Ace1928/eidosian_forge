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
def stateless_random_gamma_v2(shape: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV2_T], seed: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV2_Tseed], alpha: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV2_dtype], name=None) -> _atypes.TensorFuzzingAnnotation[TV_StatelessRandomGammaV2_dtype]:
    """Outputs deterministic pseudorandom random numbers from a gamma distribution.

  Outputs random values from a gamma distribution.

  The outputs are a deterministic function of `shape`, `seed`, and `alpha`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
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
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatelessRandomGammaV2', name, shape, seed, alpha)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateless_random_gamma_v2_eager_fallback(shape, seed, alpha, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatelessRandomGammaV2', shape=shape, seed=seed, alpha=alpha, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'T', _op._get_attr_type('T'), 'Tseed', _op._get_attr_type('Tseed'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatelessRandomGammaV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result