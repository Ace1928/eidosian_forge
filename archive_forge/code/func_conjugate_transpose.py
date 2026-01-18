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
def conjugate_transpose(x: _atypes.TensorFuzzingAnnotation[TV_ConjugateTranspose_T], perm: _atypes.TensorFuzzingAnnotation[TV_ConjugateTranspose_Tperm], name=None) -> _atypes.TensorFuzzingAnnotation[TV_ConjugateTranspose_T]:
    """Shuffle dimensions of x according to a permutation and conjugate the result.

  The output `y` has the same rank as `x`. The shapes of `x` and `y` satisfy:
    `y.shape[i] == x.shape[perm[i]] for i in [0, 1, ..., rank(x) - 1]`
    `y[i,j,k,...,s,t,u] == conj(x[perm[i], perm[j], perm[k],...,perm[s], perm[t], perm[u]])`

  Args:
    x: A `Tensor`.
    perm: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ConjugateTranspose', name, x, perm)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return conjugate_transpose_eager_fallback(x, perm, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ConjugateTranspose', x=x, perm=perm, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'Tperm', _op._get_attr_type('Tperm'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ConjugateTranspose', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result