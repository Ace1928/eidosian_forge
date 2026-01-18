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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('strings.to_hash_bucket_strong', v1=['strings.to_hash_bucket_strong', 'string_to_hash_bucket_strong'])
@deprecated_endpoints('string_to_hash_bucket_strong')
def string_to_hash_bucket_strong(input: _atypes.TensorFuzzingAnnotation[_atypes.String], num_buckets: int, key, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    """Converts each string in the input Tensor to its hash mod by a number of buckets.

  The hash function is deterministic on the content of the string within the
  process. The hash function is a keyed hash function, where attribute `key`
  defines the key of the hash function. `key` is an array of 2 elements.

  A strong hash is important when inputs may be malicious, e.g. URLs with
  additional components. Adversaries could try to make their inputs hash to the
  same bucket for a denial-of-service attack or to skew the results. A strong
  hash can be used to make it difficult to find inputs with a skewed hash value
  distribution over buckets. This requires that the hash function is
  seeded by a high-entropy (random) "key" unknown to the adversary.

  The additional robustness comes at a cost of roughly 4x higher compute
  time than `tf.string_to_hash_bucket_fast`.

  Examples:

  >>> tf.strings.to_hash_bucket_strong(["Hello", "TF"], 3, [1, 2]).numpy()
  array([2, 0])

  Args:
    input: A `Tensor` of type `string`. The strings to assign a hash bucket.
    num_buckets: An `int` that is `>= 1`. The number of buckets.
    key: A list of `ints`.
      The key used to seed the hash function, passed as a list of two uint64
      elements.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StringToHashBucketStrong', name, input, 'num_buckets', num_buckets, 'key', key)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_string_to_hash_bucket_strong((input, num_buckets, key, name), None)
            if _result is not NotImplemented:
                return _result
            return string_to_hash_bucket_strong_eager_fallback(input, num_buckets=num_buckets, key=key, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(string_to_hash_bucket_strong, (), dict(input=input, num_buckets=num_buckets, key=key, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_string_to_hash_bucket_strong((input, num_buckets, key, name), None)
        if _result is not NotImplemented:
            return _result
    num_buckets = _execute.make_int(num_buckets, 'num_buckets')
    if not isinstance(key, (list, tuple)):
        raise TypeError("Expected list for 'key' argument to 'string_to_hash_bucket_strong' Op, not %r." % key)
    key = [_execute.make_int(_i, 'key') for _i in key]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('StringToHashBucketStrong', input=input, num_buckets=num_buckets, key=key, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(string_to_hash_bucket_strong, (), dict(input=input, num_buckets=num_buckets, key=key, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('num_buckets', _op._get_attr_int('num_buckets'), 'key', _op.get_attr('key'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StringToHashBucketStrong', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result