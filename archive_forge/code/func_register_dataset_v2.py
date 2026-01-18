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
def register_dataset_v2(dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], address: _atypes.TensorFuzzingAnnotation[_atypes.String], protocol: _atypes.TensorFuzzingAnnotation[_atypes.String], external_state_policy: int, element_spec: str='', requested_dataset_id: str='', metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Registers a dataset with the tf.data service.

  Args:
    dataset: A `Tensor` of type `variant`.
    address: A `Tensor` of type `string`.
    protocol: A `Tensor` of type `string`.
    external_state_policy: An `int`.
    element_spec: An optional `string`. Defaults to `""`.
    requested_dataset_id: An optional `string`. Defaults to `""`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RegisterDatasetV2', name, dataset, address, protocol, 'external_state_policy', external_state_policy, 'element_spec', element_spec, 'requested_dataset_id', requested_dataset_id, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return register_dataset_v2_eager_fallback(dataset, address, protocol, external_state_policy=external_state_policy, element_spec=element_spec, requested_dataset_id=requested_dataset_id, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
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
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RegisterDatasetV2', dataset=dataset, address=address, protocol=protocol, external_state_policy=external_state_policy, element_spec=element_spec, requested_dataset_id=requested_dataset_id, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('external_state_policy', _op._get_attr_int('external_state_policy'), 'element_spec', _op.get_attr('element_spec'), 'requested_dataset_id', _op.get_attr('requested_dataset_id'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RegisterDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result