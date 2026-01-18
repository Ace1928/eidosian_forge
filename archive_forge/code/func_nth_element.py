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
def nth_element(input: _atypes.TensorFuzzingAnnotation[TV_NthElement_T], n: _atypes.TensorFuzzingAnnotation[_atypes.Int32], reverse: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_NthElement_T]:
    """Finds values of the `n`-th order statistic for the last dimension.

  If the input is a vector (rank-1), finds the entries which is the nth-smallest
  value in the vector and outputs their values as scalar tensor.

  For matrices (resp. higher rank input), computes the entries which is the
  nth-smallest value in each row (resp. vector along the last dimension). Thus,

      values.shape = input.shape[:-1]

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      1-D or higher with last dimension at least `n+1`.
    n: A `Tensor` of type `int32`.
      0-D. Position of sorted vector to select along the last dimension (along
      each row for matrices). Valid range of n is `[0, input.shape[:-1])`
    reverse: An optional `bool`. Defaults to `False`.
      When set to True, find the nth-largest value in the vector and vice
      versa.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'NthElement', name, input, n, 'reverse', reverse)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return nth_element_eager_fallback(input, n, reverse=reverse, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if reverse is None:
        reverse = False
    reverse = _execute.make_bool(reverse, 'reverse')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('NthElement', input=input, n=n, reverse=reverse, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('reverse', _op._get_attr_bool('reverse'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('NthElement', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result