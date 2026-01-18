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
def group_by_window_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, key_func, reduce_func, window_size_func, output_types, output_shapes, metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that computes a windowed group-by on `input_dataset`.

  // TODO(mrry): Support non-int64 keys.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    key_func_other_arguments: A list of `Tensor` objects.
    reduce_func_other_arguments: A list of `Tensor` objects.
    window_size_func_other_arguments: A list of `Tensor` objects.
    key_func: A function decorated with @Defun.
      A function mapping an element of `input_dataset`, concatenated
      with `key_func_other_arguments` to a scalar value of type DT_INT64.
    reduce_func: A function decorated with @Defun.
    window_size_func: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'GroupByWindowDataset', name, input_dataset, key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, 'key_func', key_func, 'reduce_func', reduce_func, 'window_size_func', window_size_func, 'output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return group_by_window_dataset_eager_fallback(input_dataset, key_func_other_arguments, reduce_func_other_arguments, window_size_func_other_arguments, key_func=key_func, reduce_func=reduce_func, window_size_func=window_size_func, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'group_by_window_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'group_by_window_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('GroupByWindowDataset', input_dataset=input_dataset, key_func_other_arguments=key_func_other_arguments, reduce_func_other_arguments=reduce_func_other_arguments, window_size_func_other_arguments=window_size_func_other_arguments, key_func=key_func, reduce_func=reduce_func, window_size_func=window_size_func, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('key_func', _op.get_attr('key_func'), 'reduce_func', _op.get_attr('reduce_func'), 'window_size_func', _op.get_attr('window_size_func'), 'Tkey_func_other_arguments', _op.get_attr('Tkey_func_other_arguments'), 'Treduce_func_other_arguments', _op.get_attr('Treduce_func_other_arguments'), 'Twindow_size_func_other_arguments', _op.get_attr('Twindow_size_func_other_arguments'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('GroupByWindowDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result