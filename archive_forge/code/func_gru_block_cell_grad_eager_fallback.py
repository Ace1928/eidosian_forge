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
def gru_block_cell_grad_eager_fallback(x: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], h_prev: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], w_ru: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], w_c: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], b_ru: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], b_c: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], r: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], u: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], c: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], d_h: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCellGrad_T], name, ctx):
    _attr_T, _inputs_T = _execute.args_to_matching_eager([x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h], ctx, [_dtypes.float32])
    x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h = _inputs_T
    _inputs_flat = [x, h_prev, w_ru, w_c, b_ru, b_c, r, u, c, d_h]
    _attrs = ('T', _attr_T)
    _result = _execute.execute(b'GRUBlockCellGrad', 4, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('GRUBlockCellGrad', _inputs_flat, _attrs, _result)
    _result = _GRUBlockCellGradOutput._make(_result)
    return _result