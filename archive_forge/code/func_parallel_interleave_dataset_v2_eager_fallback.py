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
def parallel_interleave_dataset_v2_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], other_arguments, cycle_length: _atypes.TensorFuzzingAnnotation[_atypes.Int64], block_length: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_parallel_calls: _atypes.TensorFuzzingAnnotation[_atypes.Int64], f, output_types, output_shapes, sloppy: bool, metadata: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'parallel_interleave_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'parallel_interleave_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if sloppy is None:
        sloppy = False
    sloppy = _execute.make_bool(sloppy, 'sloppy')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _attr_Targuments, other_arguments = _execute.convert_to_mixed_eager_tensors(other_arguments, ctx)
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    cycle_length = _ops.convert_to_tensor(cycle_length, _dtypes.int64)
    block_length = _ops.convert_to_tensor(block_length, _dtypes.int64)
    num_parallel_calls = _ops.convert_to_tensor(num_parallel_calls, _dtypes.int64)
    _inputs_flat = [input_dataset] + list(other_arguments) + [cycle_length, block_length, num_parallel_calls]
    _attrs = ('f', f, 'Targuments', _attr_Targuments, 'output_types', output_types, 'output_shapes', output_shapes, 'sloppy', sloppy, 'metadata', metadata)
    _result = _execute.execute(b'ParallelInterleaveDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ParallelInterleaveDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result