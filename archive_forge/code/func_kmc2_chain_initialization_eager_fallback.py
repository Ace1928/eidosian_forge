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
def kmc2_chain_initialization_eager_fallback(distances: _atypes.TensorFuzzingAnnotation[_atypes.Float32], seed: _atypes.TensorFuzzingAnnotation[_atypes.Int64], name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    distances = _ops.convert_to_tensor(distances, _dtypes.float32)
    seed = _ops.convert_to_tensor(seed, _dtypes.int64)
    _inputs_flat = [distances, seed]
    _attrs = None
    _result = _execute.execute(b'KMC2ChainInitialization', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('KMC2ChainInitialization', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result