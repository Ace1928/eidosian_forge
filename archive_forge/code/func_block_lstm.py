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
def block_lstm(seq_len_max: _atypes.TensorFuzzingAnnotation[_atypes.Int64], x: _atypes.TensorFuzzingAnnotation[TV_BlockLSTM_T], cs_prev: _atypes.TensorFuzzingAnnotation[TV_BlockLSTM_T], h_prev: _atypes.TensorFuzzingAnnotation[TV_BlockLSTM_T], w: _atypes.TensorFuzzingAnnotation[TV_BlockLSTM_T], wci: _atypes.TensorFuzzingAnnotation[TV_BlockLSTM_T], wcf: _atypes.TensorFuzzingAnnotation[TV_BlockLSTM_T], wco: _atypes.TensorFuzzingAnnotation[TV_BlockLSTM_T], b: _atypes.TensorFuzzingAnnotation[TV_BlockLSTM_T], forget_bias: float=1, cell_clip: float=3, use_peephole: bool=False, name=None):
    """Computes the LSTM cell forward propagation for all the time steps.

  This is equivalent to applying LSTMBlockCell in a loop, like so:

  ```python
  for x1 in unpack(x):
    i1, cs1, f1, o1, ci1, co1, h1 = LSTMBlock(
      x1, cs_prev, h_prev, w, wci, wcf, wco, b)
    cs_prev = cs1
    h_prev = h1
    i.append(i1)
    cs.append(cs1)
    f.append(f1)
    o.append(o1)
    ci.append(ci1)
    co.append(co1)
    h.append(h1)
  return pack(i), pack(cs), pack(f), pack(o), pack(ci), pack(ch), pack(h)
  ```

  Args:
    seq_len_max: A `Tensor` of type `int64`.
      Maximum time length actually used by this input. Outputs are padded
      with zeros beyond this length.
    x: A `Tensor`. Must be one of the following types: `half`, `float32`.
      The sequence input to the LSTM, shape (timelen, batch_size, num_inputs).
    cs_prev: A `Tensor`. Must have the same type as `x`.
      Value of the initial cell state.
    h_prev: A `Tensor`. Must have the same type as `x`.
      Initial output of cell (to be used for peephole).
    w: A `Tensor`. Must have the same type as `x`. The weight matrix.
    wci: A `Tensor`. Must have the same type as `x`.
      The weight matrix for input gate peephole connection.
    wcf: A `Tensor`. Must have the same type as `x`.
      The weight matrix for forget gate peephole connection.
    wco: A `Tensor`. Must have the same type as `x`.
      The weight matrix for output gate peephole connection.
    b: A `Tensor`. Must have the same type as `x`. The bias vector.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    cell_clip: An optional `float`. Defaults to `3`.
      Value to clip the 'cs' value to.
    use_peephole: An optional `bool`. Defaults to `False`.
      Whether to use peephole weights.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (i, cs, f, o, ci, co, h).

    i: A `Tensor`. Has the same type as `x`.
    cs: A `Tensor`. Has the same type as `x`.
    f: A `Tensor`. Has the same type as `x`.
    o: A `Tensor`. Has the same type as `x`.
    ci: A `Tensor`. Has the same type as `x`.
    co: A `Tensor`. Has the same type as `x`.
    h: A `Tensor`. Has the same type as `x`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BlockLSTM', name, seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, 'forget_bias', forget_bias, 'cell_clip', cell_clip, 'use_peephole', use_peephole)
            _result = _BlockLSTMOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return block_lstm_eager_fallback(seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b, forget_bias=forget_bias, cell_clip=cell_clip, use_peephole=use_peephole, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if forget_bias is None:
        forget_bias = 1
    forget_bias = _execute.make_float(forget_bias, 'forget_bias')
    if cell_clip is None:
        cell_clip = 3
    cell_clip = _execute.make_float(cell_clip, 'cell_clip')
    if use_peephole is None:
        use_peephole = False
    use_peephole = _execute.make_bool(use_peephole, 'use_peephole')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BlockLSTM', seq_len_max=seq_len_max, x=x, cs_prev=cs_prev, h_prev=h_prev, w=w, wci=wci, wcf=wcf, wco=wco, b=b, forget_bias=forget_bias, cell_clip=cell_clip, use_peephole=use_peephole, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('forget_bias', _op.get_attr('forget_bias'), 'cell_clip', _op.get_attr('cell_clip'), 'use_peephole', _op._get_attr_bool('use_peephole'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BlockLSTM', _inputs_flat, _attrs, _result)
    _result = _BlockLSTMOutput._make(_result)
    return _result