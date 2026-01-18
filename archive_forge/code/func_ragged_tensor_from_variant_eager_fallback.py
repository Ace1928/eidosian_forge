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
def ragged_tensor_from_variant_eager_fallback(encoded_ragged: _atypes.TensorFuzzingAnnotation[_atypes.Variant], input_ragged_rank: int, output_ragged_rank: int, Tvalues: TV_RaggedTensorFromVariant_Tvalues, Tsplits: TV_RaggedTensorFromVariant_Tsplits, name, ctx):
    input_ragged_rank = _execute.make_int(input_ragged_rank, 'input_ragged_rank')
    output_ragged_rank = _execute.make_int(output_ragged_rank, 'output_ragged_rank')
    Tvalues = _execute.make_type(Tvalues, 'Tvalues')
    if Tsplits is None:
        Tsplits = _dtypes.int64
    Tsplits = _execute.make_type(Tsplits, 'Tsplits')
    encoded_ragged = _ops.convert_to_tensor(encoded_ragged, _dtypes.variant)
    _inputs_flat = [encoded_ragged]
    _attrs = ('input_ragged_rank', input_ragged_rank, 'output_ragged_rank', output_ragged_rank, 'Tvalues', Tvalues, 'Tsplits', Tsplits)
    _result = _execute.execute(b'RaggedTensorFromVariant', output_ragged_rank + 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RaggedTensorFromVariant', _inputs_flat, _attrs, _result)
    _result = [_result[:output_ragged_rank]] + _result[output_ragged_rank:]
    _result = _RaggedTensorFromVariantOutput._make(_result)
    return _result