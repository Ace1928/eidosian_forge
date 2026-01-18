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
def string_to_hash_bucket_strong_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.String], num_buckets: int, key, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    if not isinstance(key, (list, tuple)):
        raise TypeError("Expected list for 'key' argument to 'string_to_hash_bucket_strong' Op, not %r." % key)
    key = [_execute.make_int(_i, 'key') for _i in key]
    input = _ops.convert_to_tensor(input, _dtypes.string)
    _inputs_flat = [input]
    _attrs = ('num_buckets', num_buckets, 'key', key)
    _result = _execute.execute(b'StringToHashBucketStrong', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('StringToHashBucketStrong', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result