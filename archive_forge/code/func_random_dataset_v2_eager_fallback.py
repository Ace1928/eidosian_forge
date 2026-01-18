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
def random_dataset_v2_eager_fallback(seed: _atypes.TensorFuzzingAnnotation[_atypes.Int64], seed2: _atypes.TensorFuzzingAnnotation[_atypes.Int64], seed_generator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], output_types, output_shapes, rerandomize_each_iteration: bool, metadata: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'random_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'random_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if rerandomize_each_iteration is None:
        rerandomize_each_iteration = False
    rerandomize_each_iteration = _execute.make_bool(rerandomize_each_iteration, 'rerandomize_each_iteration')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    seed = _ops.convert_to_tensor(seed, _dtypes.int64)
    seed2 = _ops.convert_to_tensor(seed2, _dtypes.int64)
    seed_generator = _ops.convert_to_tensor(seed_generator, _dtypes.resource)
    _inputs_flat = [seed, seed2, seed_generator]
    _attrs = ('rerandomize_each_iteration', rerandomize_each_iteration, 'output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata)
    _result = _execute.execute(b'RandomDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RandomDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result