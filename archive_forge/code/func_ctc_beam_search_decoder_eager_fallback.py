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
def ctc_beam_search_decoder_eager_fallback(inputs: _atypes.TensorFuzzingAnnotation[TV_CTCBeamSearchDecoder_T], sequence_length: _atypes.TensorFuzzingAnnotation[_atypes.Int32], beam_width: int, top_paths: int, merge_repeated: bool, name, ctx):
    beam_width = _execute.make_int(beam_width, 'beam_width')
    top_paths = _execute.make_int(top_paths, 'top_paths')
    if merge_repeated is None:
        merge_repeated = True
    merge_repeated = _execute.make_bool(merge_repeated, 'merge_repeated')
    _attr_T, (inputs,) = _execute.args_to_matching_eager([inputs], ctx, [_dtypes.float32, _dtypes.float64], _dtypes.float32)
    sequence_length = _ops.convert_to_tensor(sequence_length, _dtypes.int32)
    _inputs_flat = [inputs, sequence_length]
    _attrs = ('beam_width', beam_width, 'top_paths', top_paths, 'merge_repeated', merge_repeated, 'T', _attr_T)
    _result = _execute.execute(b'CTCBeamSearchDecoder', top_paths + top_paths + top_paths + 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('CTCBeamSearchDecoder', _inputs_flat, _attrs, _result)
    _result = [_result[:top_paths]] + _result[top_paths:]
    _result = _result[:1] + [_result[1:1 + top_paths]] + _result[1 + top_paths:]
    _result = _result[:2] + [_result[2:2 + top_paths]] + _result[2 + top_paths:]
    _result = _CTCBeamSearchDecoderOutput._make(_result)
    return _result