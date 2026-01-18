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
def dataset_to_graph_v2_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], external_state_policy: int, strip_device_assignment: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if external_state_policy is None:
        external_state_policy = 0
    external_state_policy = _execute.make_int(external_state_policy, 'external_state_policy')
    if strip_device_assignment is None:
        strip_device_assignment = False
    strip_device_assignment = _execute.make_bool(strip_device_assignment, 'strip_device_assignment')
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset]
    _attrs = ('external_state_policy', external_state_policy, 'strip_device_assignment', strip_device_assignment)
    _result = _execute.execute(b'DatasetToGraphV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DatasetToGraphV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result