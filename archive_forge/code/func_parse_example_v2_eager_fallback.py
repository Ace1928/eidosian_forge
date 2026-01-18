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
def parse_example_v2_eager_fallback(serialized: _atypes.TensorFuzzingAnnotation[_atypes.String], names: _atypes.TensorFuzzingAnnotation[_atypes.String], sparse_keys: _atypes.TensorFuzzingAnnotation[_atypes.String], dense_keys: _atypes.TensorFuzzingAnnotation[_atypes.String], ragged_keys: _atypes.TensorFuzzingAnnotation[_atypes.String], dense_defaults, num_sparse: int, sparse_types, ragged_value_types, ragged_split_types, dense_shapes, name, ctx):
    num_sparse = _execute.make_int(num_sparse, 'num_sparse')
    if not isinstance(sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'sparse_types' argument to 'parse_example_v2' Op, not %r." % sparse_types)
    sparse_types = [_execute.make_type(_t, 'sparse_types') for _t in sparse_types]
    if not isinstance(ragged_value_types, (list, tuple)):
        raise TypeError("Expected list for 'ragged_value_types' argument to 'parse_example_v2' Op, not %r." % ragged_value_types)
    ragged_value_types = [_execute.make_type(_t, 'ragged_value_types') for _t in ragged_value_types]
    if not isinstance(ragged_split_types, (list, tuple)):
        raise TypeError("Expected list for 'ragged_split_types' argument to 'parse_example_v2' Op, not %r." % ragged_split_types)
    ragged_split_types = [_execute.make_type(_t, 'ragged_split_types') for _t in ragged_split_types]
    if not isinstance(dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'dense_shapes' argument to 'parse_example_v2' Op, not %r." % dense_shapes)
    dense_shapes = [_execute.make_shape(_s, 'dense_shapes') for _s in dense_shapes]
    _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
    serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
    names = _ops.convert_to_tensor(names, _dtypes.string)
    sparse_keys = _ops.convert_to_tensor(sparse_keys, _dtypes.string)
    dense_keys = _ops.convert_to_tensor(dense_keys, _dtypes.string)
    ragged_keys = _ops.convert_to_tensor(ragged_keys, _dtypes.string)
    _inputs_flat = [serialized, names, sparse_keys, dense_keys, ragged_keys] + list(dense_defaults)
    _attrs = ('Tdense', _attr_Tdense, 'num_sparse', num_sparse, 'sparse_types', sparse_types, 'ragged_value_types', ragged_value_types, 'ragged_split_types', ragged_split_types, 'dense_shapes', dense_shapes)
    _result = _execute.execute(b'ParseExampleV2', num_sparse + len(sparse_types) + num_sparse + len(dense_defaults) + len(ragged_value_types) + len(ragged_split_types), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ParseExampleV2', _inputs_flat, _attrs, _result)
    _result = [_result[:num_sparse]] + _result[num_sparse:]
    _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
    _result = _result[:2] + [_result[2:2 + num_sparse]] + _result[2 + num_sparse:]
    _result = _result[:3] + [_result[3:3 + len(dense_defaults)]] + _result[3 + len(dense_defaults):]
    _result = _result[:4] + [_result[4:4 + len(ragged_value_types)]] + _result[4 + len(ragged_value_types):]
    _result = _result[:5] + [_result[5:]]
    _result = _ParseExampleV2Output._make(_result)
    return _result