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
def crop_and_resize_grad_image_eager_fallback(grads: _atypes.TensorFuzzingAnnotation[_atypes.Float32], boxes: _atypes.TensorFuzzingAnnotation[_atypes.Float32], box_ind: _atypes.TensorFuzzingAnnotation[_atypes.Int32], image_size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], T: TV_CropAndResizeGradImage_T, method: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_CropAndResizeGradImage_T]:
    T = _execute.make_type(T, 'T')
    if method is None:
        method = 'bilinear'
    method = _execute.make_str(method, 'method')
    grads = _ops.convert_to_tensor(grads, _dtypes.float32)
    boxes = _ops.convert_to_tensor(boxes, _dtypes.float32)
    box_ind = _ops.convert_to_tensor(box_ind, _dtypes.int32)
    image_size = _ops.convert_to_tensor(image_size, _dtypes.int32)
    _inputs_flat = [grads, boxes, box_ind, image_size]
    _attrs = ('T', T, 'method', method)
    _result = _execute.execute(b'CropAndResizeGradImage', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CropAndResizeGradImage', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result