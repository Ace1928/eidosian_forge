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
def ctc_greedy_decoder_eager_fallback(inputs: _atypes.TensorFuzzingAnnotation[TV_CTCGreedyDecoder_T], sequence_length: _atypes.TensorFuzzingAnnotation[_atypes.Int32], merge_repeated: bool, blank_index: int, name, ctx):
    if merge_repeated is None:
        merge_repeated = False
    merge_repeated = _execute.make_bool(merge_repeated, 'merge_repeated')
    if blank_index is None:
        blank_index = -1
    blank_index = _execute.make_int(blank_index, 'blank_index')
    _attr_T, (inputs,) = _execute.args_to_matching_eager([inputs], ctx, [_dtypes.float32, _dtypes.float64], _dtypes.float32)
    sequence_length = _ops.convert_to_tensor(sequence_length, _dtypes.int32)
    _inputs_flat = [inputs, sequence_length]
    _attrs = ('merge_repeated', merge_repeated, 'blank_index', blank_index, 'T', _attr_T)
    _result = _execute.execute(b'CTCGreedyDecoder', 4, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CTCGreedyDecoder', _inputs_flat, _attrs, _result)
    _result = _CTCGreedyDecoderOutput._make(_result)
    return _result