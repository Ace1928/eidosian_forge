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
def dataset_to_graph_v2(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], external_state_policy: int=0, strip_device_assignment: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Returns a serialized GraphDef representing `input_dataset`.

  Returns a graph representation for `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the dataset to return the graph representation for.
    external_state_policy: An optional `int`. Defaults to `0`.
    strip_device_assignment: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DatasetToGraphV2', name, input_dataset, 'external_state_policy', external_state_policy, 'strip_device_assignment', strip_device_assignment)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return dataset_to_graph_v2_eager_fallback(input_dataset, external_state_policy=external_state_policy, strip_device_assignment=strip_device_assignment, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if external_state_policy is None:
        external_state_policy = 0
    external_state_policy = _execute.make_int(external_state_policy, 'external_state_policy')
    if strip_device_assignment is None:
        strip_device_assignment = False
    strip_device_assignment = _execute.make_bool(strip_device_assignment, 'strip_device_assignment')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DatasetToGraphV2', input_dataset=input_dataset, external_state_policy=external_state_policy, strip_device_assignment=strip_device_assignment, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('external_state_policy', _op._get_attr_int('external_state_policy'), 'strip_device_assignment', _op._get_attr_bool('strip_device_assignment'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DatasetToGraphV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result