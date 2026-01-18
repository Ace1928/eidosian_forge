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
def fused_resize_and_pad_conv2d_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_FusedResizeAndPadConv2D_T], size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], paddings: _atypes.TensorFuzzingAnnotation[_atypes.Int32], filter: _atypes.TensorFuzzingAnnotation[TV_FusedResizeAndPadConv2D_T], mode: str, strides, padding: str, resize_align_corners: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_FusedResizeAndPadConv2D_T]:
    mode = _execute.make_str(mode, 'mode')
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'fused_resize_and_pad_conv2d' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if resize_align_corners is None:
        resize_align_corners = False
    resize_align_corners = _execute.make_bool(resize_align_corners, 'resize_align_corners')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.half, _dtypes.float32, _dtypes.float64])
    input, filter = _inputs_T
    size = _ops.convert_to_tensor(size, _dtypes.int32)
    paddings = _ops.convert_to_tensor(paddings, _dtypes.int32)
    _inputs_flat = [input, size, paddings, filter]
    _attrs = ('T', _attr_T, 'resize_align_corners', resize_align_corners, 'mode', mode, 'strides', strides, 'padding', padding)
    _result = _execute.execute(b'FusedResizeAndPadConv2D', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FusedResizeAndPadConv2D', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result