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
def batch_self_adjoint_eig_v2(input: _atypes.TensorFuzzingAnnotation[TV_BatchSelfAdjointEigV2_T], compute_v: bool=True, name=None):
    """TODO: add doc.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
    compute_v: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (e, v).

    e: A `Tensor`. Has the same type as `input`.
    v: A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BatchSelfAdjointEigV2', name, input, 'compute_v', compute_v)
            _result = _BatchSelfAdjointEigV2Output._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return batch_self_adjoint_eig_v2_eager_fallback(input, compute_v=compute_v, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if compute_v is None:
        compute_v = True
    compute_v = _execute.make_bool(compute_v, 'compute_v')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BatchSelfAdjointEigV2', input=input, compute_v=compute_v, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('compute_v', _op._get_attr_bool('compute_v'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BatchSelfAdjointEigV2', _inputs_flat, _attrs, _result)
    _result = _BatchSelfAdjointEigV2Output._make(_result)
    return _result