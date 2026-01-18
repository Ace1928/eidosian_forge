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
def parallel_concat(values: List[_atypes.TensorFuzzingAnnotation[TV_ParallelConcat_T]], shape, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ParallelConcat_T]:
    """Concatenates a list of `N` tensors along the first dimension.

  The input tensors are all required to have size 1 in the first dimension.

  For example:

  ```
  # 'x' is [[1, 4]]
  # 'y' is [[2, 5]]
  # 'z' is [[3, 6]]
  parallel_concat([x, y, z]) => [[1, 4], [2, 5], [3, 6]]  # Pack along first dim.
  ```

  The difference between concat and parallel_concat is that concat requires all
  of the inputs be computed before the operation will begin but doesn't require
  that the input shapes be known during graph construction.  Parallel concat
  will copy pieces of the input into the output as they become available, in
  some situations this can provide a performance benefit.

  Args:
    values: A list of at least 1 `Tensor` objects with the same type.
      Tensors to be concatenated. All must have size 1 in the first dimension
      and same shape.
    shape: A `tf.TensorShape` or list of `ints`.
      the final shape of the result; should be equal to the shapes of any input
      but with the number of input values in the first dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `values`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ParallelConcat', name, values, 'shape', shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return parallel_concat_eager_fallback(values, shape=shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(values, (list, tuple)):
        raise TypeError("Expected list for 'values' argument to 'parallel_concat' Op, not %r." % values)
    _attr_N = len(values)
    shape = _execute.make_shape(shape, 'shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ParallelConcat', values=values, shape=shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('N', _op._get_attr_int('N'), 'T', _op._get_attr_type('T'), 'shape', _op.get_attr('shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ParallelConcat', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result