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
def boosted_trees_quantile_stream_resource_add_summaries_eager_fallback(quantile_stream_resource_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], summaries: List[_atypes.TensorFuzzingAnnotation[_atypes.Float32]], name, ctx):
    if not isinstance(summaries, (list, tuple)):
        raise TypeError("Expected list for 'summaries' argument to 'boosted_trees_quantile_stream_resource_add_summaries' Op, not %r." % summaries)
    _attr_num_features = len(summaries)
    quantile_stream_resource_handle = _ops.convert_to_tensor(quantile_stream_resource_handle, _dtypes.resource)
    summaries = _ops.convert_n_to_tensor(summaries, _dtypes.float32)
    _inputs_flat = [quantile_stream_resource_handle] + list(summaries)
    _attrs = ('num_features', _attr_num_features)
    _result = _execute.execute(b'BoostedTreesQuantileStreamResourceAddSummaries', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result