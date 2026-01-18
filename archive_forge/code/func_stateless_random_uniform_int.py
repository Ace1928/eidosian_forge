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
def stateless_random_uniform_int(shape: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomUniformInt_T], seed: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomUniformInt_Tseed], minval: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomUniformInt_dtype], maxval: _atypes.TensorFuzzingAnnotation[TV_StatelessRandomUniformInt_dtype], name=None) -> _atypes.TensorFuzzingAnnotation[TV_StatelessRandomUniformInt_dtype]:
    """Outputs deterministic pseudorandom random integers from a uniform distribution.

  The generated values follow a uniform distribution in the range `[minval, maxval)`.

  The outputs are a deterministic function of `shape`, `seed`, `minval`, and `maxval`.

  Args:
    shape: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      The shape of the output tensor.
    seed: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2 seeds (shape [2]).
    minval: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Minimum value (inclusive, scalar).
    maxval: A `Tensor`. Must have the same type as `minval`.
      Maximum value (exclusive, scalar).
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `minval`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatelessRandomUniformInt', name, shape, seed, minval, maxval)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateless_random_uniform_int_eager_fallback(shape, seed, minval, maxval, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatelessRandomUniformInt', shape=shape, seed=seed, minval=minval, maxval=maxval, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'T', _op._get_attr_type('T'), 'Tseed', _op._get_attr_type('Tseed'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatelessRandomUniformInt', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result