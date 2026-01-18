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
def list_diff(x: _atypes.TensorFuzzingAnnotation[TV_ListDiff_T], y: _atypes.TensorFuzzingAnnotation[TV_ListDiff_T], out_idx: TV_ListDiff_out_idx=_dtypes.int32, name=None):
    """Computes the difference between two lists of numbers or strings.

  Given a list `x` and a list `y`, this operation returns a list `out` that
  represents all values that are in `x` but not in `y`. The returned list `out`
  is sorted in the same order that the numbers appear in `x` (duplicates are
  preserved). This operation also returns a list `idx` that represents the
  position of each `out` element in `x`. In other words:

  `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

  For example, given this input:

  ```
  x = [1, 2, 3, 4, 5, 6]
  y = [1, 3, 5]
  ```

  This operation would return:

  ```
  out ==> [2, 4, 6]
  idx ==> [1, 3, 5]
  ```

  Args:
    x: A `Tensor`. 1-D. Values to keep.
    y: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
    out_idx: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int32`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (out, idx).

    out: A `Tensor`. Has the same type as `x`.
    idx: A `Tensor` of type `out_idx`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ListDiff', name, x, y, 'out_idx', out_idx)
            _result = _ListDiffOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return list_diff_eager_fallback(x, y, out_idx=out_idx, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if out_idx is None:
        out_idx = _dtypes.int32
    out_idx = _execute.make_type(out_idx, 'out_idx')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ListDiff', x=x, y=y, out_idx=out_idx, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'out_idx', _op._get_attr_type('out_idx'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ListDiff', _inputs_flat, _attrs, _result)
    _result = _ListDiffOutput._make(_result)
    return _result