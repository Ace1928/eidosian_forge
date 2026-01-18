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
def stats_aggregator_set_summary_writer_eager_fallback(stats_aggregator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], summary: _atypes.TensorFuzzingAnnotation[_atypes.Resource], name, ctx):
    stats_aggregator = _ops.convert_to_tensor(stats_aggregator, _dtypes.resource)
    summary = _ops.convert_to_tensor(summary, _dtypes.resource)
    _inputs_flat = [stats_aggregator, summary]
    _attrs = None
    _result = _execute.execute(b'StatsAggregatorSetSummaryWriter', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result