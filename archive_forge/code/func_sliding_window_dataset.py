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
def sliding_window_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], window_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], window_shift: _atypes.TensorFuzzingAnnotation[_atypes.Int64], window_stride: _atypes.TensorFuzzingAnnotation[_atypes.Int64], output_types, output_shapes, drop_remainder: bool=True, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that passes a sliding window over `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    window_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements in the
      sliding window.
    window_shift: A `Tensor` of type `int64`.
      A scalar representing the steps moving the sliding window
      forward in one iteration. It must be positive.
    window_stride: A `Tensor` of type `int64`.
      A scalar representing the stride of the input elements of the sliding window.
      It must be positive.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    drop_remainder: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SlidingWindowDataset', name, input_dataset, window_size, window_shift, window_stride, 'drop_remainder', drop_remainder, 'output_types', output_types, 'output_shapes', output_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return sliding_window_dataset_eager_fallback(input_dataset, window_size, window_shift, window_stride, drop_remainder=drop_remainder, output_types=output_types, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'sliding_window_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'sliding_window_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if drop_remainder is None:
        drop_remainder = True
    drop_remainder = _execute.make_bool(drop_remainder, 'drop_remainder')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SlidingWindowDataset', input_dataset=input_dataset, window_size=window_size, window_shift=window_shift, window_stride=window_stride, output_types=output_types, output_shapes=output_shapes, drop_remainder=drop_remainder, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('drop_remainder', _op._get_attr_bool('drop_remainder'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SlidingWindowDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result