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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('math.invert_permutation', v1=['math.invert_permutation', 'invert_permutation'])
@deprecated_endpoints('invert_permutation')
def invert_permutation(x: _atypes.TensorFuzzingAnnotation[TV_InvertPermutation_T], name=None) -> _atypes.TensorFuzzingAnnotation[TV_InvertPermutation_T]:
    """Computes the inverse permutation of a tensor.

  This operation computes the inverse of an index permutation. It takes a 1-D
  integer tensor `x`, which represents the indices of a zero-based array, and
  swaps each value with its index position. In other words, for an output tensor
  `y` and an input tensor `x`, this operation computes the following:

  `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

  The values must include 0. There can be no duplicate values or negative values.

  For example:

  ```
  # tensor `x` is [3, 4, 0, 2, 1]
  invert_permutation(x) ==> [2, 4, 3, 0, 1]
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`. 1-D.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'InvertPermutation', name, x)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_invert_permutation((x, name), None)
            if _result is not NotImplemented:
                return _result
            return invert_permutation_eager_fallback(x, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(invert_permutation, (), dict(x=x, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_invert_permutation((x, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('InvertPermutation', x=x, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(invert_permutation, (), dict(x=x, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('InvertPermutation', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result