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
def anonymous_seed_generator(seed: _atypes.TensorFuzzingAnnotation[_atypes.Int64], seed2: _atypes.TensorFuzzingAnnotation[_atypes.Int64], reshuffle: _atypes.TensorFuzzingAnnotation[_atypes.Bool], name=None):
    """TODO: add doc.

  Args:
    seed: A `Tensor` of type `int64`.
    seed2: A `Tensor` of type `int64`.
    reshuffle: A `Tensor` of type `bool`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (handle, deleter).

    handle: A `Tensor` of type `resource`.
    deleter: A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AnonymousSeedGenerator', name, seed, seed2, reshuffle)
            _result = _AnonymousSeedGeneratorOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return anonymous_seed_generator_eager_fallback(seed, seed2, reshuffle, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('AnonymousSeedGenerator', seed=seed, seed2=seed2, reshuffle=reshuffle, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('AnonymousSeedGenerator', _inputs_flat, _attrs, _result)
    _result = _AnonymousSeedGeneratorOutput._make(_result)
    return _result