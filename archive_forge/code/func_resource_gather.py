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
def resource_gather(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], indices: _atypes.TensorFuzzingAnnotation[TV_ResourceGather_Tindices], dtype: TV_ResourceGather_dtype, batch_dims: int=0, validate_indices: bool=True, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ResourceGather_dtype]:
    """Gather slices from the variable pointed to by `resource` according to `indices`.

  `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
  Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

  ```python
      # Scalar indices
      output[:, ..., :] = params[indices, :, ... :]

      # Vector indices
      output[i, :, ..., :] = params[indices[i], :, ... :]

      # Higher rank indices
      output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]
  ```

  Args:
    resource: A `Tensor` of type `resource`.
    indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    dtype: A `tf.DType`.
    batch_dims: An optional `int`. Defaults to `0`.
    validate_indices: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ResourceGather', name, resource, indices, 'batch_dims', batch_dims, 'validate_indices', validate_indices, 'dtype', dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return resource_gather_eager_fallback(resource, indices, batch_dims=batch_dims, validate_indices=validate_indices, dtype=dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    if batch_dims is None:
        batch_dims = 0
    batch_dims = _execute.make_int(batch_dims, 'batch_dims')
    if validate_indices is None:
        validate_indices = True
    validate_indices = _execute.make_bool(validate_indices, 'validate_indices')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ResourceGather', resource=resource, indices=indices, dtype=dtype, batch_dims=batch_dims, validate_indices=validate_indices, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('batch_dims', _op._get_attr_int('batch_dims'), 'validate_indices', _op._get_attr_bool('validate_indices'), 'dtype', _op._get_attr_type('dtype'), 'Tindices', _op._get_attr_type('Tindices'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ResourceGather', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result