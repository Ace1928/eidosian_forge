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
def extract_jpeg_shape_eager_fallback(contents: _atypes.TensorFuzzingAnnotation[_atypes.String], output_type: TV_ExtractJpegShape_output_type, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ExtractJpegShape_output_type]:
    if output_type is None:
        output_type = _dtypes.int32
    output_type = _execute.make_type(output_type, 'output_type')
    contents = _ops.convert_to_tensor(contents, _dtypes.string)
    _inputs_flat = [contents]
    _attrs = ('output_type', output_type)
    _result = _execute.execute(b'ExtractJpegShape', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ExtractJpegShape', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result