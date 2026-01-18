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
def parse_single_example_eager_fallback(serialized: _atypes.TensorFuzzingAnnotation[_atypes.String], dense_defaults, num_sparse: int, sparse_keys, dense_keys, sparse_types, dense_shapes, name, ctx):
    num_sparse = _execute.make_int(num_sparse, 'num_sparse')
    if not isinstance(sparse_keys, (list, tuple)):
        raise TypeError("Expected list for 'sparse_keys' argument to 'parse_single_example' Op, not %r." % sparse_keys)
    sparse_keys = [_execute.make_str(_s, 'sparse_keys') for _s in sparse_keys]
    if not isinstance(dense_keys, (list, tuple)):
        raise TypeError("Expected list for 'dense_keys' argument to 'parse_single_example' Op, not %r." % dense_keys)
    dense_keys = [_execute.make_str(_s, 'dense_keys') for _s in dense_keys]
    if not isinstance(sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'sparse_types' argument to 'parse_single_example' Op, not %r." % sparse_types)
    sparse_types = [_execute.make_type(_t, 'sparse_types') for _t in sparse_types]
    if not isinstance(dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'dense_shapes' argument to 'parse_single_example' Op, not %r." % dense_shapes)
    dense_shapes = [_execute.make_shape(_s, 'dense_shapes') for _s in dense_shapes]
    _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
    serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
    _inputs_flat = [serialized] + list(dense_defaults)
    _attrs = ('num_sparse', num_sparse, 'sparse_keys', sparse_keys, 'dense_keys', dense_keys, 'sparse_types', sparse_types, 'Tdense', _attr_Tdense, 'dense_shapes', dense_shapes)
    _result = _execute.execute(b'ParseSingleExample', num_sparse + len(sparse_types) + num_sparse + len(dense_defaults), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ParseSingleExample', _inputs_flat, _attrs, _result)
    _result = [_result[:num_sparse]] + _result[num_sparse:]
    _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
    _result = _result[:2] + [_result[2:2 + num_sparse]] + _result[2 + num_sparse:]
    _result = _result[:3] + [_result[3:]]
    _result = _ParseSingleExampleOutput._make(_result)
    return _result