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
def extract_image_patches_eager_fallback(images: _atypes.TensorFuzzingAnnotation[TV_ExtractImagePatches_T], ksizes, strides, rates, padding: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ExtractImagePatches_T]:
    if not isinstance(ksizes, (list, tuple)):
        raise TypeError("Expected list for 'ksizes' argument to 'extract_image_patches' Op, not %r." % ksizes)
    ksizes = [_execute.make_int(_i, 'ksizes') for _i in ksizes]
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'extract_image_patches' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    if not isinstance(rates, (list, tuple)):
        raise TypeError("Expected list for 'rates' argument to 'extract_image_patches' Op, not %r." % rates)
    rates = [_execute.make_int(_i, 'rates') for _i in rates]
    padding = _execute.make_str(padding, 'padding')
    _attr_T, (images,) = _execute.args_to_matching_eager([images], ctx, [_dtypes.bfloat16, _dtypes.half, _dtypes.float32, _dtypes.float64, _dtypes.int8, _dtypes.int16, _dtypes.int32, _dtypes.int64, _dtypes.uint8, _dtypes.uint16, _dtypes.uint32, _dtypes.uint64, _dtypes.complex64, _dtypes.complex128, _dtypes.bool])
    _inputs_flat = [images]
    _attrs = ('ksizes', ksizes, 'strides', strides, 'rates', rates, 'T', _attr_T, 'padding', padding)
    _result = _execute.execute(b'ExtractImagePatches', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ExtractImagePatches', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result