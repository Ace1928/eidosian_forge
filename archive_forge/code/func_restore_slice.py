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
def restore_slice(file_pattern: _atypes.TensorFuzzingAnnotation[_atypes.String], tensor_name: _atypes.TensorFuzzingAnnotation[_atypes.String], shape_and_slice: _atypes.TensorFuzzingAnnotation[_atypes.String], dt: TV_RestoreSlice_dt, preferred_shard: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[TV_RestoreSlice_dt]:
    """Restores a tensor from checkpoint files.

  This is like `Restore` except that restored tensor can be listed as filling
  only a slice of a larger tensor.  `shape_and_slice` specifies the shape of the
  larger tensor and the slice that the restored tensor covers.

  The `shape_and_slice` input has the same format as the
  elements of the `shapes_and_slices` input of the `SaveSlices` op.

  Args:
    file_pattern: A `Tensor` of type `string`.
      Must have a single element. The pattern of the files from
      which we read the tensor.
    tensor_name: A `Tensor` of type `string`.
      Must have a single element. The name of the tensor to be
      restored.
    shape_and_slice: A `Tensor` of type `string`.
      Scalar. The shapes and slice specifications to use when
      restoring a tensors.
    dt: A `tf.DType`. The type of the tensor to be restored.
    preferred_shard: An optional `int`. Defaults to `-1`.
      Index of file to open first if multiple files match
      `file_pattern`. See the documentation for `Restore`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dt`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RestoreSlice', name, file_pattern, tensor_name, shape_and_slice, 'dt', dt, 'preferred_shard', preferred_shard)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return restore_slice_eager_fallback(file_pattern, tensor_name, shape_and_slice, dt=dt, preferred_shard=preferred_shard, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dt = _execute.make_type(dt, 'dt')
    if preferred_shard is None:
        preferred_shard = -1
    preferred_shard = _execute.make_int(preferred_shard, 'preferred_shard')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RestoreSlice', file_pattern=file_pattern, tensor_name=tensor_name, shape_and_slice=shape_and_slice, dt=dt, preferred_shard=preferred_shard, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dt', _op._get_attr_type('dt'), 'preferred_shard', _op._get_attr_int('preferred_shard'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RestoreSlice', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result