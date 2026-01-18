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
def boosted_trees_quantile_stream_resource_deserialize_eager_fallback(quantile_stream_resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], bucket_boundaries: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], name, ctx):
    if not isinstance(bucket_boundaries, (list, tuple)):
        raise TypeError("Expected list for 'bucket_boundaries' argument to 'boosted_trees_quantile_stream_resource_deserialize' Op, not %r." % bucket_boundaries)
    _attr_num_streams = len(bucket_boundaries)
    quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
    bucket_boundaries = _ops.convert_n_to_tensor(bucket_boundaries, _dtypes.float32)
    _inputs_flat = [quantile_stream_resource_handle] + list(bucket_boundaries)
    _attrs = ('num_streams', _attr_num_streams)
    _result = _execute.execute(b'BoostedTreesQuantileStreamResourceDeserialize', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result