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
def register_dataset_v2_eager_fallback(dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], address: _atypes.TensorFuzzingAnnotation[_atypes.String], protocol: _atypes.TensorFuzzingAnnotation[_atypes.String], external_state_policy: int, element_spec: str, requested_dataset_id: str, metadata: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    external_state_policy = _execute.make_int(external_state_policy, 'external_state_policy')
    if element_spec is None:
        element_spec = ''
    element_spec = _execute.make_str(element_spec, 'element_spec')
    if requested_dataset_id is None:
        requested_dataset_id = ''
    requested_dataset_id = _execute.make_str(requested_dataset_id, 'requested_dataset_id')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
    address = _ops.convert_to_tensor(address, _dtypes.string)
    protocol = _ops.convert_to_tensor(protocol, _dtypes.string)
    _inputs_flat = [dataset, address, protocol]
    _attrs = ('external_state_policy', external_state_policy, 'element_spec', element_spec, 'requested_dataset_id', requested_dataset_id, 'metadata', metadata)
    _result = _execute.execute(b'RegisterDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RegisterDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result