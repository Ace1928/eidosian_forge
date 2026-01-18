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
def fixed_length_record_dataset_v2_eager_fallback(filenames: _atypes.TensorFuzzingAnnotation[_atypes.String], header_bytes: _atypes.TensorFuzzingAnnotation[_atypes.Int64], record_bytes: _atypes.TensorFuzzingAnnotation[_atypes.Int64], footer_bytes: _atypes.TensorFuzzingAnnotation[_atypes.Int64], buffer_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], compression_type: _atypes.TensorFuzzingAnnotation[_atypes.String], metadata: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
    header_bytes = _ops.convert_to_tensor(header_bytes, _dtypes.int64)
    record_bytes = _ops.convert_to_tensor(record_bytes, _dtypes.int64)
    footer_bytes = _ops.convert_to_tensor(footer_bytes, _dtypes.int64)
    buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
    compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
    _inputs_flat = [filenames, header_bytes, record_bytes, footer_bytes, buffer_size, compression_type]
    _attrs = ('metadata', metadata)
    _result = _execute.execute(b'FixedLengthRecordDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FixedLengthRecordDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result