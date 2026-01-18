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
def fake_quant_with_min_max_vars_gradient_eager_fallback(gradients: _atypes.TensorFuzzingAnnotation[_atypes.Float32], inputs: _atypes.TensorFuzzingAnnotation[_atypes.Float32], min: _atypes.TensorFuzzingAnnotation[_atypes.Float32], max: _atypes.TensorFuzzingAnnotation[_atypes.Float32], num_bits: int, narrow_range: bool, name, ctx):
    if num_bits is None:
        num_bits = 8
    num_bits = _execute.make_int(num_bits, 'num_bits')
    if narrow_range is None:
        narrow_range = False
    narrow_range = _execute.make_bool(narrow_range, 'narrow_range')
    gradients = _ops.convert_to_tensor(gradients, _dtypes.float32)
    inputs = _ops.convert_to_tensor(inputs, _dtypes.float32)
    min = _ops.convert_to_tensor(min, _dtypes.float32)
    max = _ops.convert_to_tensor(max, _dtypes.float32)
    _inputs_flat = [gradients, inputs, min, max]
    _attrs = ('num_bits', num_bits, 'narrow_range', narrow_range)
    _result = _execute.execute(b'FakeQuantWithMinMaxVarsGradient', 3, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('FakeQuantWithMinMaxVarsGradient', _inputs_flat, _attrs, _result)
    _result = _FakeQuantWithMinMaxVarsGradientOutput._make(_result)
    return _result