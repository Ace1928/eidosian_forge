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
def group_by_window_dataset_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, key_func, reduce_func, window_size_func, output_types, output_shapes, metadata: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'group_by_window_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'group_by_window_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _attr_Tkey_func_other_arguments, key_func_other_arguments = _execute.convert_to_mixed_eager_tensors(key_func_other_arguments, ctx)
    _attr_Treduce_func_other_arguments, reduce_func_other_arguments = _execute.convert_to_mixed_eager_tensors(reduce_func_other_arguments, ctx)
    _attr_Twindow_size_func_other_arguments, window_size_func_other_arguments = _execute.convert_to_mixed_eager_tensors(window_size_func_other_arguments, ctx)
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset] + list(key_func_other_arguments) + list(reduce_func_other_arguments) + list(window_size_func_other_arguments)
    _attrs = ('key_func', key_func, 'reduce_func', reduce_func, 'window_size_func', window_size_func, 'Tkey_func_other_arguments', _attr_Tkey_func_other_arguments, 'Treduce_func_other_arguments', _attr_Treduce_func_other_arguments, 'Twindow_size_func_other_arguments', _attr_Twindow_size_func_other_arguments, 'output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata)
    _result = _execute.execute(b'GroupByWindowDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('GroupByWindowDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result