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
def generate_bounding_box_proposals_eager_fallback(scores: _atypes.TensorFuzzingAnnotation[_atypes.Float32], bbox_deltas: _atypes.TensorFuzzingAnnotation[_atypes.Float32], image_info: _atypes.TensorFuzzingAnnotation[_atypes.Float32], anchors: _atypes.TensorFuzzingAnnotation[_atypes.Float32], nms_threshold: _atypes.TensorFuzzingAnnotation[_atypes.Float32], pre_nms_topn: _atypes.TensorFuzzingAnnotation[_atypes.Int32], min_size: _atypes.TensorFuzzingAnnotation[_atypes.Float32], post_nms_topn: int, name, ctx):
    if post_nms_topn is None:
        post_nms_topn = 300
    post_nms_topn = _execute.make_int(post_nms_topn, 'post_nms_topn')
    scores = _ops.convert_to_tensor(scores, _dtypes.float32)
    bbox_deltas = _ops.convert_to_tensor(bbox_deltas, _dtypes.float32)
    image_info = _ops.convert_to_tensor(image_info, _dtypes.float32)
    anchors = _ops.convert_to_tensor(anchors, _dtypes.float32)
    nms_threshold = _ops.convert_to_tensor(nms_threshold, _dtypes.float32)
    pre_nms_topn = _ops.convert_to_tensor(pre_nms_topn, _dtypes.int32)
    min_size = _ops.convert_to_tensor(min_size, _dtypes.float32)
    _inputs_flat = [scores, bbox_deltas, image_info, anchors, nms_threshold, pre_nms_topn, min_size]
    _attrs = ('post_nms_topn', post_nms_topn)
    _result = _execute.execute(b'GenerateBoundingBoxProposals', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('GenerateBoundingBoxProposals', _inputs_flat, _attrs, _result)
    _result = _GenerateBoundingBoxProposalsOutput._make(_result)
    return _result