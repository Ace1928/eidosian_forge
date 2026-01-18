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
def max_pool3d_grad(orig_input: _atypes.TensorFuzzingAnnotation[TV_MaxPool3DGrad_TInput], orig_output: _atypes.TensorFuzzingAnnotation[TV_MaxPool3DGrad_TInput], grad: _atypes.TensorFuzzingAnnotation[TV_MaxPool3DGrad_T], ksize, strides, padding: str, data_format: str='NDHWC', name=None) -> _atypes.TensorFuzzingAnnotation[TV_MaxPool3DGrad_T]:
    """Computes gradients of 3D max pooling function.

  Args:
    orig_input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      The original input tensor.
    orig_output: A `Tensor`. Must have the same type as `orig_input`.
      The original output tensor.
    grad: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      Output backprop of shape `[batch, depth, rows, cols, channels]`.
    ksize: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The size of the window for each dimension of
      the input tensor. Must have `ksize[0] = ksize[4] = 1`.
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
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grad`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'MaxPool3DGrad', name, orig_input, orig_output, grad, 'ksize', ksize, 'strides', strides, 'padding', padding, 'data_format', data_format)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return max_pool3d_grad_eager_fallback(orig_input, orig_output, grad, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(ksize, (list, tuple)):
        raise TypeError("Expected list for 'ksize' argument to 'max_pool3d_grad' Op, not %r." % ksize)
    ksize = [_execute.make_int(_i, 'ksize') for _i in ksize]
    if not isinstance(strides, (list, tuple)):
        raise TypeError("Expected list for 'strides' argument to 'max_pool3d_grad' Op, not %r." % strides)
    strides = [_execute.make_int(_i, 'strides') for _i in strides]
    padding = _execute.make_str(padding, 'padding')
    if data_format is None:
        data_format = 'NDHWC'
    data_format = _execute.make_str(data_format, 'data_format')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('MaxPool3DGrad', orig_input=orig_input, orig_output=orig_output, grad=grad, ksize=ksize, strides=strides, padding=padding, data_format=data_format, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('ksize', _op.get_attr('ksize'), 'strides', _op.get_attr('strides'), 'padding', _op.get_attr('padding'), 'data_format', _op.get_attr('data_format'), 'T', _op._get_attr_type('T'), 'TInput', _op._get_attr_type('TInput'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('MaxPool3DGrad', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result