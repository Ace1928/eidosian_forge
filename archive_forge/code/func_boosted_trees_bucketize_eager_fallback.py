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
def boosted_trees_bucketize_eager_fallback(float_values: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], bucket_boundaries: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], name, ctx):
    if not isinstance(float_values, (list, tuple)):
        raise TypeError("Expected list for 'float_values' argument to 'boosted_trees_bucketize' Op, not %r." % float_values)
    _attr_num_features = len(float_values)
    if not isinstance(bucket_boundaries, (list, tuple)):
        raise TypeError("Expected list for 'bucket_boundaries' argument to 'boosted_trees_bucketize' Op, not %r." % bucket_boundaries)
    if len(bucket_boundaries) != _attr_num_features:
        raise ValueError("List argument 'bucket_boundaries' to 'boosted_trees_bucketize' Op with length %d must match length %d of argument 'float_values'." % (len(bucket_boundaries), _attr_num_features))
    float_values = _ops.convert_n_to_tensor(float_values, _dtypes.float32)
    bucket_boundaries = _ops.convert_n_to_tensor(bucket_boundaries, _dtypes.float32)
    _inputs_flat = list(float_values) + list(bucket_boundaries)
    _attrs = ('num_features', _attr_num_features)
    _result = _execute.execute(b'BoostedTreesBucketize', _attr_num_features, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('BoostedTreesBucketize', _inputs_flat, _attrs, _result)
    return _result