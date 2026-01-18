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
def parse_example_eager_fallback(serialized: _atypes.TensorFuzzingAnnotation[_atypes.String], names: _atypes.TensorFuzzingAnnotation[_atypes.String], sparse_keys: List[_atypes.TensorFuzzingAnnotation[_atypes.String]], dense_keys: List[_atypes.TensorFuzzingAnnotation[_atypes.String]], dense_defaults, sparse_types, dense_shapes, name, ctx):
    if not isinstance(sparse_keys, (list, tuple)):
        raise TypeError("Expected list for 'sparse_keys' argument to 'parse_example' Op, not %r." % sparse_keys)
    _attr_Nsparse = len(sparse_keys)
    if not isinstance(dense_keys, (list, tuple)):
        raise TypeError("Expected list for 'dense_keys' argument to 'parse_example' Op, not %r." % dense_keys)
    _attr_Ndense = len(dense_keys)
    if not isinstance(sparse_types, (list, tuple)):
        raise TypeError("Expected list for 'sparse_types' argument to 'parse_example' Op, not %r." % sparse_types)
    sparse_types = [_execute.make_type(_t, 'sparse_types') for _t in sparse_types]
    if not isinstance(dense_shapes, (list, tuple)):
        raise TypeError("Expected list for 'dense_shapes' argument to 'parse_example' Op, not %r." % dense_shapes)
    dense_shapes = [_execute.make_shape(_s, 'dense_shapes') for _s in dense_shapes]
    _attr_Tdense, dense_defaults = _execute.convert_to_mixed_eager_tensors(dense_defaults, ctx)
    serialized = _ops.convert_to_tensor(serialized, _dtypes.string)
    names = _ops.convert_to_tensor(names, _dtypes.string)
    sparse_keys = _ops.convert_n_to_tensor(sparse_keys, _dtypes.string)
    dense_keys = _ops.convert_n_to_tensor(dense_keys, _dtypes.string)
    _inputs_flat = [serialized, names] + list(sparse_keys) + list(dense_keys) + list(dense_defaults)
    _attrs = ('Nsparse', _attr_Nsparse, 'Ndense', _attr_Ndense, 'sparse_types', sparse_types, 'Tdense', _attr_Tdense, 'dense_shapes', dense_shapes)
    _result = _execute.execute(b'ParseExample', _attr_Nsparse + len(sparse_types) + _attr_Nsparse + len(dense_defaults), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ParseExample', _inputs_flat, _attrs, _result)
    _result = [_result[:_attr_Nsparse]] + _result[_attr_Nsparse:]
    _result = _result[:1] + [_result[1:1 + len(sparse_types)]] + _result[1 + len(sparse_types):]
    _result = _result[:2] + [_result[2:2 + _attr_Nsparse]] + _result[2 + _attr_Nsparse:]
    _result = _result[:3] + [_result[3:]]
    _result = _ParseExampleOutput._make(_result)
    return _result