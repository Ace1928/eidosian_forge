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
def infeed_enqueue_prelinearized_buffer_eager_fallback(input: _atypes.TensorFuzzingAnnotation[_atypes.Variant], device_ordinal: int, name, ctx):
    if device_ordinal is None:
        device_ordinal = -1
    device_ordinal = _execute.make_int(device_ordinal, 'device_ordinal')
    input = _ops.convert_to_tensor(input, _dtypes.variant)
    _inputs_flat = [input]
    _attrs = ('device_ordinal', device_ordinal)
    _result = _execute.execute(b'InfeedEnqueuePrelinearizedBuffer', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result