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
def conv3d_backprop_input_v2(input_sizes: _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropInputV2_Tshape], filter: _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropInputV2_T], out_backprop: _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropInputV2_T], strides, padding: str, data_format: str='NDHWC', dilations=[1, 1, 1, 1, 1], name=None) -> _atypes.TensorFuzzingAnnotation[TV_Conv3DBackpropInputV2_T]:
    """Computes the gradients of 3-D convolution with respect to the input.

  Args:
    input_sizes: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      An integer vector representing the tensor shape of `input`,
      where `input` is a 5-D
      `[batch, depth, rows, cols, in_channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`, `float64`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    data_format: An optional `string` from: `"NDHWC", "NCDHW"`. Defaults to `"NDHWC"`.
      The data format of the input and output data. With the
      default format "NDHWC", the data is stored in the order of:
          [batch, in_depth, in_height, in_width, in_channels].
      Alternatively, the format could be "NCDHW", the data storage order is:
          [batch, in_channels, in_depth, in_height, in_width].
    dilations: An optional list of `ints`. Defaults to `[1, 1, 1, 1, 1]`.
      1-D tensor of length 5.  The dilation factor for each dimension of
      `input`. If set to k > 1, there will be k-1 skipped cells between each
      filter element on that dimension. The dimension order is determined by the
      value of `data_format`, see above for details. Dilations in the batch and
      depth dimensions must be 1.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Conv3DBackpropInputV2', name, input_sizes, filter, out_backprop, 'strides', strides, 'padding', padding, 'data_format', data_format, 'dilations', dilations)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return conv3d_backprop_input_v2_eager_fallback(input_sizes, filter, out_backprop, strides=strides, padding=padding, data_format=data_format, dilations=dilations, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'conv3d_backprop_input_v2' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if data_format is None:
        data_format = 'NDHWC'
    data_format = _execute.make_str(data_format, 'data_format')
    if dilations is None:
        dilations = [1, 1, 1, 1, 1]
    if not isinstance(dilations, (list, tuple)):
        raise TypeError("Expected list for 'dilations' argument to 'conv3d_backprop_input_v2' Op, not %r." % dilations)
    dilations = [_execute.make_int(_i, 'dilations') for _i in dilations]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Conv3DBackpropInputV2', input_sizes=input_sizes, filter=filter, out_backprop=out_backprop, strides=strides, padding=padding, data_format=data_format, dilations=dilations, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'strides', _op.get_attr('strides'), 'padding', _op.get_attr('padding'), 'data_format', _op.get_attr('data_format'), 'dilations', _op.get_attr('dilations'), 'Tshape', _op._get_attr_type('Tshape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Conv3DBackpropInputV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result