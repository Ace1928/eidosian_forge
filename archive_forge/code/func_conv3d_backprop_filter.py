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
def conv3d_backprop_filter(input: _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropFilter_T], filter: _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropFilter_T], out_backprop: _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropFilter_T], strides, padding: str, dilations=[1, 1, 1, 1, 1], name=None) -> _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropFilter_T]:
    """Computes the gradients of 3-D convolution with respect to the filter.

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
      Shape `[batch, depth, rows, cols, in_channels]`.
    filter: A `Tensor`. Must have the same type as `input`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `input`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Conv3DBackpropFilter', name, input, filter, out_backprop, 'strides', strides, 'padding', padding, 'dilations', dilations)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return conv3d_backprop_filter_eager_fallback(input, filter, out_backprop, strides=strides, padding=padding, dilations=dilations, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'conv3d_backprop_filter' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if dilations is None:
        dilations = [1, 1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
        raise TypeError("Expected list for 'dilations' argument to 'conv3d_backprop_filter' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, 'dilations') for _i in dilations]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Conv3DBackpropFilter', input=input, filter=filter, out_backprop=out_backprop, strides=strides, padding=padding, dilations=dilations, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'strides', _op.get_attr('strides'), 'padding', _op.get_attr('padding'), 'dilations', _op.get_attr('dilations'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Conv3DBackpropFilter', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result