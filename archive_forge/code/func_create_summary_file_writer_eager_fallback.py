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
def create_summary_file_writer_eager_fallback(writer: _atypes.TensorFuzzingAnnotation[_atypes.Resource], logdir: _atypes.TensorFuzzingAnnotation[_atypes.String], max_queue: _atypes.TensorFuzzingAnnotation[_atypes.Int32], flush_millis: _atypes.TensorFuzzingAnnotation[_atypes.Int32], filename_suffix: _atypes.TensorFuzzingAnnotation[_atypes.String], name, ctx):
    writer = _ops.convert_to_tensor(writer, _dtypes.resource)
    logdir = _ops.convert_to_tensor(logdir, _dtypes.string)
    max_queue = _ops.convert_to_tensor(max_queue, _dtypes.int32)
    flush_millis = _ops.convert_to_tensor(flush_millis, _dtypes.int32)
    filename_suffix = _ops.convert_to_tensor(filename_suffix, _dtypes.string)
    _inputs_flat = [writer, logdir, max_queue, flush_millis, filename_suffix]
    _attrs = None
    _result = _execute.execute(b'CreateSummaryFileWriter', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result