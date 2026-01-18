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
def compute_accidental_hits_eager_fallback(true_classes: _atypes.TensorFuzzingAnnotation[_atypes.Int64], sampled_candidates: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_true: int, seed: int, seed2: int, name, ctx):
    num_true = _execute.make_int(num_true, 'num_true')
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    true_classes = _ops.convert_to_tensor(true_classes, _dtypes.int64)
    sampled_candidates = _ops.convert_to_tensor(sampled_candidates, _dtypes.int64)
    _inputs_flat = [true_classes, sampled_candidates]
    _attrs = ('num_true', num_true, 'seed', seed, 'seed2', seed2)
    _result = _execute.execute(b'ComputeAccidentalHits', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ComputeAccidentalHits', _inputs_flat, _attrs, _result)
    _result = _ComputeAccidentalHitsOutput._make(_result)
    return _result