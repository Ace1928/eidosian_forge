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
def range_dataset_eager_fallback(start: _atypes.TensorFuzzingAnnotation[_atypes.Int64], stop: _atypes.TensorFuzzingAnnotation[_atypes.Int64], step: _atypes.TensorFuzzingAnnotation[_atypes.Int64], output_types, output_shapes, metadata: str, replicate_on_split: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'range_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'range_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    if replicate_on_split is None:
        replicate_on_split = False
    replicate_on_split = _execute.make_bool(replicate_on_split, 'replicate_on_split')
    start = _ops.convert_to_tensor(start, _dtypes.int64)
    stop = _ops.convert_to_tensor(stop, _dtypes.int64)
    step = _ops.convert_to_tensor(step, _dtypes.int64)
    _inputs_flat = [start, stop, step]
    _attrs = ('output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata, 'replicate_on_split', replicate_on_split)
    _result = _execute.execute(b'RangeDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RangeDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result