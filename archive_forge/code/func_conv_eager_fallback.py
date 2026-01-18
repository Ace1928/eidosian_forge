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
def conv_eager_fallback(input: _atypes.TensorFuzzingAnnotation[TV_Conv_T], filter: _atypes.TensorFuzzingAnnotation[TV_Conv_T], strides, padding: str, explicit_paddings, data_format: str, dilations, batch_dims: int, groups: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_Conv_T]:
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'conv' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if explicit_paddings is None:
        explicit_paddings = []
    if not isinstance(explicit_paddings, (list, tuple)):
        raise TypeError("Expected list for 'explicit_paddings' argument to 'conv' Op, not %r." % explicit_paddings)
    explicit_paddings = [_execute.make_int(_i, 'explicit_paddings') for _i in explicit_paddings]
    if data_format is None:
        data_format = 'CHANNELS_LAST'
    data_format = _execute.make_str(data_format, 'data_format')
    if dilations is None:
        dilations = []
    if not isinstance(dilations, (list, tuple)):
        raise TypeError("Expected list for 'dilations' argument to 'conv' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, 'dilations') for _i in dilations]
    if batch_dims is None:
        batch_dims = 1
    batch_dims = _execute.make_int(batch_dims, 'batch_dims')
    if groups is None:
        groups = 1
    groups = _execute.make_int(groups, 'groups')
    _attr_T, _inputs_T = _execute.args_to_matching_eager([input, filter], ctx, [_dtypes.half, _dtypes.bfloat16, _dtypes.float32, _dtypes.float64, _dtypes.int32])
    input, filter = _inputs_T
    _inputs_flat = [input, filter]
    _attrs = ('T', _attr_T, 'strides', strides, 'padding', padding, 'explicit_paddings', explicit_paddings, 'data_format', data_format, 'dilations', dilations, 'batch_dims', batch_dims, 'groups', groups)
    _result = _execute.execute(b'Conv', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('Conv', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result