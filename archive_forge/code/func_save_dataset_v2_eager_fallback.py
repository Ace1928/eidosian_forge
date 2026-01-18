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
def save_dataset_v2_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], path: _atypes.TensorFuzzingAnnotation[_atypes.String], shard_func_other_args, shard_func, output_types, output_shapes, compression: str, use_shard_func: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'save_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'save_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if compression is None:
        compression = ''
    compression = _execute.make_str(compression, 'compression')
    if use_shard_func is None:
        use_shard_func = True
    use_shard_func = _execute.make_bool(use_shard_func, 'use_shard_func')
    _attr_Tshard_func_args, shard_func_other_args = _execute.convert_to_mixed_eager_tensors(shard_func_other_args, ctx)
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    path = _ops.convert_to_tensor(path, _dtypes.string)
    _inputs_flat = [input_dataset, path] + list(shard_func_other_args)
    _attrs = ('compression', compression, 'shard_func', shard_func, 'use_shard_func', use_shard_func, 'Tshard_func_args', _attr_Tshard_func_args, 'output_types', output_types, 'output_shapes', output_shapes)
    _result = _execute.execute(b'SaveDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SaveDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result