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
def immutable_const(dtype: TV_ImmutableConst_dtype, shape, memory_region_name: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ImmutableConst_dtype]:
    """Returns immutable tensor from memory region.

  The current implementation memmaps the tensor from a file.

  Args:
    dtype: A `tf.DType`. Type of the returned tensor.
    shape: A `tf.TensorShape` or list of `ints`. Shape of the returned tensor.
    memory_region_name: A `string`.
      Name of readonly memory region used by the tensor, see
      NewReadOnlyMemoryRegionFromFile in tensorflow::Env.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ImmutableConst', name, 'dtype', dtype, 'shape', shape, 'memory_region_name', memory_region_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return immutable_const_eager_fallback(dtype=dtype, shape=shape, memory_region_name=memory_region_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    shape = _execute.make_shape(shape, 'shape')
    memory_region_name = _execute.make_str(memory_region_name, 'memory_region_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ImmutableConst', dtype=dtype, shape=shape, memory_region_name=memory_region_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'shape', _op.get_attr('shape'), 'memory_region_name', _op.get_attr('memory_region_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ImmutableConst', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result