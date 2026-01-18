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
def ragged_bincount(splits: _atypes.TensorFuzzingAnnotation[_atypes.Int64], values: _atypes.TensorFuzzingAnnotation[TV_RaggedBincount_Tidx], size: _atypes.TensorFuzzingAnnotation[TV_RaggedBincount_Tidx], weights: _atypes.TensorFuzzingAnnotation[TV_RaggedBincount_T], binary_output: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[TV_RaggedBincount_T]:
    """Counts the number of occurrences of each value in an integer array.

  Outputs a vector with length `size` and the same dtype as `weights`. If
  `weights` are empty, then index `i` stores the number of times the value `i` is
  counted in `arr`. If `weights` are non-empty, then index `i` stores the sum of
  the value in `weights` at each index where the corresponding value in `arr` is
  `i`.

  Values in `arr` outside of the range [0, size) are ignored.

  Args:
    splits: A `Tensor` of type `int64`. 1D int64 `Tensor`.
    values: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      2D int `Tensor`.
    size: A `Tensor`. Must have the same type as `values`.
      non-negative int scalar `Tensor`.
    weights: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      is an int32, int64, float32, or float64 `Tensor` with the same
      shape as `input`, or a length-0 `Tensor`, in which case it acts as all weights
      equal to 1.
    binary_output: An optional `bool`. Defaults to `False`.
      bool; Whether the kernel should count the appearance or number of occurrences.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `weights`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedBincount', name, splits, values, size, weights, 'binary_output', binary_output)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return ragged_bincount_eager_fallback(splits, values, size, weights, binary_output=binary_output, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if binary_output is None:
        binary_output = False
    binary_output = _execute.make_bool(binary_output, 'binary_output')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedBincount', splits=splits, values=values, size=size, weights=weights, binary_output=binary_output, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tidx', _op._get_attr_type('Tidx'), 'T', _op._get_attr_type('T'), 'binary_output', _op._get_attr_bool('binary_output'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedBincount', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result