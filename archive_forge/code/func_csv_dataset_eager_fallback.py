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
def csv_dataset_eager_fallback(filenames: _atypes.TensorFuzzingAnnotation[_atypes.String], compression_type: _atypes.TensorFuzzingAnnotation[_atypes.String], buffer_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], header: _atypes.TensorFuzzingAnnotation[_atypes.Bool], field_delim: _atypes.TensorFuzzingAnnotation[_atypes.String], use_quote_delim: _atypes.TensorFuzzingAnnotation[_atypes.Bool], na_value: _atypes.TensorFuzzingAnnotation[_atypes.String], select_cols: _atypes.TensorFuzzingAnnotation[_atypes.Int64], record_defaults, output_shapes, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'csv_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _attr_output_types, record_defaults = _execute.convert_to_mixed_eager_tensors(record_defaults, ctx)
    filenames = _ops.convert_to_tensor(filenames, _dtypes.string)
    compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
    buffer_size = _ops.convert_to_tensor(buffer_size, _dtypes.int64)
    header = _ops.convert_to_tensor(header, _dtypes.bool)
    field_delim = _ops.convert_to_tensor(field_delim, _dtypes.string)
    use_quote_delim = _ops.convert_to_tensor(use_quote_delim, _dtypes.bool)
    na_value = _ops.convert_to_tensor(na_value, _dtypes.string)
    select_cols = _ops.convert_to_tensor(select_cols, _dtypes.int64)
    _inputs_flat = [filenames, compression_type, buffer_size, header, field_delim, use_quote_delim, na_value, select_cols] + list(record_defaults)
    _attrs = ('output_types', _attr_output_types, 'output_shapes', output_shapes)
    _result = _execute.execute(b'CSVDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CSVDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result