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
def gru_block_cell(x: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCell_T], h_prev: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCell_T], w_ru: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCell_T], w_c: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCell_T], b_ru: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCell_T], b_c: _atypes.TensorFuzzingAnnotation[TV_GRUBlockCell_T], name=None):
    """Computes the GRU cell forward propagation for 1 time step.

  Args
      x: Input to the GRU cell.
      h_prev: State input from the previous GRU cell.
      w_ru: Weight matrix for the reset and update gate.
      w_c: Weight matrix for the cell connection gate.
      b_ru: Bias vector for the reset and update gate.
      b_c: Bias vector for the cell connection gate.

  Returns
      r: Output of the reset gate.
      u: Output of the update gate.
      c: Output of the cell connection gate.
      h: Current state of the GRU cell.

  Note on notation of the variables:

  Concatenation of a and b is represented by a_b
  Element-wise dot product of a and b is represented by ab
  Element-wise dot product is represented by \\circ
  Matrix multiplication is represented by *

  Biases are initialized with :
  `b_ru` - constant_initializer(1.0)
  `b_c` - constant_initializer(0.0)

  This kernel op implements the following mathematical equations:

  ```
  x_h_prev = [x, h_prev]

  [r_bar u_bar] = x_h_prev * w_ru + b_ru

  r = sigmoid(r_bar)
  u = sigmoid(u_bar)

  h_prevr = h_prev \\circ r

  x_h_prevr = [x h_prevr]

  c_bar = x_h_prevr * w_c + b_c
  c = tanh(c_bar)

  h = (1-u) \\circ c + u \\circ h_prev
  ```

  Args:
    x: A `Tensor`. Must be one of the following types: `float32`.
    h_prev: A `Tensor`. Must have the same type as `x`.
    w_ru: A `Tensor`. Must have the same type as `x`.
    w_c: A `Tensor`. Must have the same type as `x`.
    b_ru: A `Tensor`. Must have the same type as `x`.
    b_c: A `Tensor`. Must have the same type as `x`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (r, u, c, h).

    r: A `Tensor`. Has the same type as `x`.
    u: A `Tensor`. Has the same type as `x`.
    c: A `Tensor`. Has the same type as `x`.
    h: A `Tensor`. Has the same type as `x`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'GRUBlockCell', name, x, h_prev, w_ru, w_c, b_ru, b_c)
            _result = _GRUBlockCellOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return gru_block_cell_eager_fallback(x, h_prev, w_ru, w_c, b_ru, b_c, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('GRUBlockCell', x=x, h_prev=h_prev, w_ru=w_ru, w_c=w_c, b_ru=b_ru, b_c=b_c, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('GRUBlockCell', _inputs_flat, _attrs, _result)
    _result = _GRUBlockCellOutput._make(_result)
    return _result