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
def model_dataset_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], output_types, output_shapes, algorithm: int, cpu_budget: int, ram_budget: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'model_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'model_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if algorithm is None:
        algorithm = 0
    algorithm = _execute.make_int(algorithm, 'algorithm')
    if cpu_budget is None:
        cpu_budget = 0
    cpu_budget = _execute.make_int(cpu_budget, 'cpu_budget')
    if ram_budget is None:
        ram_budget = 0
    ram_budget = _execute.make_int(ram_budget, 'ram_budget')
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset]
    _attrs = ('algorithm', algorithm, 'cpu_budget', cpu_budget, 'ram_budget', ram_budget, 'output_types', output_types, 'output_shapes', output_shapes)
    _result = _execute.execute(b'ModelDataset', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ModelDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result