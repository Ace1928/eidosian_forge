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
def resource_apply_centered_rms_prop_eager_fallback(var: _atypes.TensorFuzzingAnnotation[_atypes.Resource], mg: _atypes.TensorFuzzingAnnotation[_atypes.Resource], ms: _atypes.TensorFuzzingAnnotation[_atypes.Resource], mom: _atypes.TensorFuzzingAnnotation[_atypes.Resource], lr: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], rho: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], momentum: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], epsilon: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], grad: _atypes.TensorFuzzingAnnotation[TV_ResourceApplyCenteredRMSProp_T], use_locking: bool, name, ctx):
    if use_locking is None:
        use_locking = False
    use_locking = _execute.make_bool(use_locking, 'use_locking')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([lr, rho, momentum, epsilon, grad], ctx, [_dtypes.float32, _dtypes.float64, _dtypes.int32, _dtypes.uint8, _dtypes.int16, _dtypes.int8, _dtypes.complex64, _dtypes.int64, _dtypes.qint8, _dtypes.quint8, _dtypes.qint32, _dtypes.bfloat16, _dtypes.qint16, _dtypes.quint16, _dtypes.uint16, _dtypes.complex128, _dtypes.half, _dtypes.uint32, _dtypes.uint64])
    lr, rho, momentum, epsilon, grad = _inputs_T
    var = _ops.convert_to_tensor(var, _dtypes.resource)
    mg = _ops.convert_to_tensor(mg, _dtypes.resource)
    ms = _ops.convert_to_tensor(ms, _dtypes.resource)
    mom = _ops.convert_to_tensor(mom, _dtypes.resource)
    _inputs_flat = [var, mg, ms, mom, lr, rho, momentum, epsilon, grad]
    _attrs = ('T', _attr_T, 'use_locking', use_locking)
    _result = _execute.execute(b'ResourceApplyCenteredRMSProp', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result