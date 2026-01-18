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
def outfeed_dequeue_v2_eager_fallback(device_ordinal: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_OutfeedDequeueV2_dtype, shape, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_OutfeedDequeueV2_dtype]:
    dtype = _execute.make_type(dtype, 'dtype')
    shape = _execute.make_shape(shape, 'shape')
    device_ordinal = _ops.convert_to_tensor(device_ordinal, _dtypes.int32)
    _inputs_flat = [device_ordinal]
    _attrs = ('dtype', dtype, 'shape', shape)
    _result = _execute.execute(b'OutfeedDequeueV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('OutfeedDequeueV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result