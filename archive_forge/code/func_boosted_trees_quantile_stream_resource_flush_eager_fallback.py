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
def boosted_trees_quantile_stream_resource_flush_eager_fallback(quantile_stream_resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], num_buckets: _atypes.TensorFuzzingAnnotation[_atypes.Int64], generate_quantiles: bool, name, ctx):
    if generate_quantiles is None:
        generate_quantiles = False
    generate_quantiles = _execute.make_bool(generate_quantiles, 'generate_quantiles')
    quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
    num_buckets = _ops.convert_to_tensor(num_buckets, _dtypes.int64)
    _inputs_flat = [quantile_stream_resource_handle, num_buckets]
    _attrs = ('generate_quantiles', generate_quantiles)
    _result = _execute.execute(b'BoostedTreesQuantileStreamResourceFlush', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result