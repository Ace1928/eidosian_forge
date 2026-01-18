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
def experimental_map_and_batch_dataset_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], other_arguments, batch_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_parallel_calls: _atypes.TensorFuzzingAnnotation[_atypes.Int64], drop_remainder: _atypes.TensorFuzzingAnnotation[_atypes.Bool], f, output_types, output_shapes, preserve_cardinality: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'experimental_map_and_batch_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'experimental_map_and_batch_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if preserve_cardinality is None:
        preserve_cardinality = False
    preserve_cardinality = _execute.make_bool(preserve_cardinality, 'preserve_cardinality')
    _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    batch_size = _ops.convert_to_tensor(batch_size, _dtypes.int64)
    num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
    drop_remainder = _ops.convert_to_tensor(drop_remainder, _dtypes.bool)
    _inputs_flat = [input_dataset] + list(other_arguments) + [batch_size, num_parallel_calls, drop_remainder]
    _attrs = ('f', f, 'Targuments', _attr_Targuments, 'output_types', output_types, 'output_shapes', output_shapes, 'preserve_cardinality', preserve_cardinality)
    _result = _execute.execute(b'ExperimentalMapAndBatchDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ExperimentalMapAndBatchDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result