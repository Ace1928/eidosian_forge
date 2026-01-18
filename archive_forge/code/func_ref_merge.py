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
def ref_merge(inputs: List[_atypes.TensorFuzzingAnnotation[TV_RefMerge_T]], name=None):
    """Forwards the value of an available tensor from `inputs` to `output`.

  `Merge` waits for at least one of the tensors in `inputs` to become available.
  It is usually combined with `Switch` to implement branching.

  `Merge` forwards the first tensor for become available to `output`, and sets
  `value_index` to its index in `inputs`.

  Args:
    inputs: A list of at least 1 mutable `Tensor` objects with the same type.
      The input tensors, exactly one of which will become available.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, value_index).

    output: A mutable `Tensor`. Has the same type as `inputs`.
    value_index: A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("ref_merge op does not support eager execution. Arg 'output' is a ref.")
    if not isinstance(inputs, (list, tuple)):
        raise TypeError("Expected list for 'inputs' argument to 'ref_merge' Op, not %r." % inputs)
    _attr_N = len(inputs)
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RefMerge', inputs=inputs, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'N', _op._get_attr_int('N'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RefMerge', _inputs_flat, _attrs, _result)
    _result = _RefMergeOutput._make(_result)
    return _result