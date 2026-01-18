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
@tf_export('extract_volume_patches')
def extract_volume_patches(input: _atypes.TensorFuzzingAnnotation[TV_ExtractVolumePatches_T], ksizes, strides, padding: str, name=None) -> _atypes.TensorFuzzingAnnotation[TV_ExtractVolumePatches_T]:
    """Extract `patches` from `input` and put them in the `"depth"` output dimension. 3D extension of `extract_image_patches`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
      5-D Tensor with shape `[batch, in_planes, in_rows, in_cols, depth]`.
    ksizes: A list of `ints` that has length `>= 5`.
      The size of the sliding window for each dimension of `input`.
    strides: A list of `ints` that has length `>= 5`.
      1-D of length 5. How far the centers of two consecutive patches are in
      `input`. Must be: `[1, stride_planes, stride_rows, stride_cols, 1]`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.

      The size-related attributes are specified as follows:

      ```python
      ksizes = [1, ksize_planes, ksize_rows, ksize_cols, 1]
      strides = [1, stride_planes, strides_rows, strides_cols, 1]
      ```
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ExtractVolumePatches', name, input, 'ksizes', ksizes, 'strides', strides, 'padding', padding)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_extract_volume_patches((input, ksizes, strides, padding, name), None)
            if _result is not NotImplemented:
                return _result
            return extract_volume_patches_eager_fallback(input, ksizes=ksizes, strides=strides, padding=padding, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(extract_volume_patches, (), dict(input=input, ksizes=ksizes, strides=strides, padding=padding, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_extract_volume_patches((input, ksizes, strides, padding, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(ksizes, (list, tuple)):
        raise TypeError("Expected list for 'ksizes' argument to 'extract_volume_patches' Op, not %r." % ksizes)
    ksizes = [_execute.make_int(_i, 'ksizes') for _i in ksizes]
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'extract_volume_patches' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('ExtractVolumePatches', input=input, ksizes=ksizes, strides=strides, padding=padding, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(extract_volume_patches, (), dict(input=input, ksizes=ksizes, strides=strides, padding=padding, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('ksizes', _op.get_attr('ksizes'), 'strides', _op.get_attr('strides'), 'T', _op._get_attr_type('T'), 'padding', _op.get_attr('padding'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ExtractVolumePatches', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result