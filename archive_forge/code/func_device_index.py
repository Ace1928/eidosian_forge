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
def device_index(device_names, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    """Return the index of device the op runs.

  Given a list of device names, this operation returns the index of the device
  this op runs. The length of the list is returned in two cases:
  (1) Device does not exist in the given device list.
  (2) It is in XLA compilation.

  Args:
    device_names: A list of `strings`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DeviceIndex', name, 'device_names', device_names)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return device_index_eager_fallback(device_names=device_names, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(device_names, (list, tuple)):
        raise TypeError("Expected list for 'device_names' argument to 'device_index' Op, not %r." % device_names)
    device_names = [_execute.make_str(_s, 'device_names') for _s in device_names]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DeviceIndex', device_names=device_names, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('device_names', _op.get_attr('device_names'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DeviceIndex', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result