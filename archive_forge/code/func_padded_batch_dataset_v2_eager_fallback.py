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
def padded_batch_dataset_v2_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], batch_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], padded_shapes: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], padding_values, drop_remainder: _atypes.TensorFuzzingAnnotation[_atypes.Bool], output_shapes, parallel_copy: bool, metadata: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(padded_shapes, (list, tuple)):
        raise TypeError("Expected list for 'padded_shapes' argument to 'padded_batch_dataset_v2' Op, not %r." % padded_shapes)
    _attr_N = len(padded_shapes)
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'padded_batch_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if parallel_copy is None:
        parallel_copy = False
    parallel_copy = _execute.make_bool(parallel_copy, 'parallel_copy')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _attr_Toutput_types, padding_values = _execute.convert_to_mixed_eager_tensors(padding_values, ctx)
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
    padded_shapes = _ops.convert_n_to_tensor(padded_shapes, _dtypes.int64)
    drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
    _inputs_flat = [input_dataset, batch_size] + list(padded_shapes) + list(padding_values) + [drop_remainder]
    _attrs = ('parallel_copy', parallel_copy, 'Toutput_types', _attr_Toutput_types, 'output_shapes', output_shapes, 'N', _attr_N, 'metadata', metadata)
    _result = _execute.execute(b'PaddedBatchDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('PaddedBatchDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result