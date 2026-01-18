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
def directed_interleave_dataset_eager_fallback(selector_input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], data_input_datasets: List[_atypes.TensorFuzzingAnnotation[_atypes.Variant]], output_types, output_shapes, stop_on_empty_dataset: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(data_input_datasets, (list, tuple)):
        raise TypeError("Expected list for 'data_input_datasets' argument to 'directed_interleave_dataset' Op, not %r." % data_input_datasets)
    _attr_N = len(data_input_datasets)
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'directed_interleave_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'directed_interleave_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if stop_on_empty_dataset is None:
        stop_on_empty_dataset = False
    stop_on_empty_dataset = _execute.make_bool(stop_on_empty_dataset, 'stop_on_empty_dataset')
    selector_input_dataset = _ops.convert_to_tensor(selector_input_dataset, _dtypes.variant)
    data_input_datasets = _ops.convert_n_to_tensor(data_input_datasets, _dtypes.variant)
    _inputs_flat = [selector_input_dataset] + list(data_input_datasets)
    _attrs = ('output_types', output_types, 'output_shapes', output_shapes, 'N', _attr_N, 'stop_on_empty_dataset', stop_on_empty_dataset)
    _result = _execute.execute(b'DirectedInterleaveDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DirectedInterleaveDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result