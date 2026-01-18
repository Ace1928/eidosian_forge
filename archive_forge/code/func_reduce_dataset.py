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
def reduce_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], initial_state, other_arguments, f, output_types, output_shapes, use_inter_op_parallelism: bool=True, metadata: str='', name=None):
    """Reduces the input dataset to a singleton using a reduce function.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    initial_state: A list of `Tensor` objects.
      A nested structure of tensors, representing the initial state of the
      transformation.
    other_arguments: A list of `Tensor` objects.
    f: A function decorated with @Defun.
      A function that maps `(old_state, input_element)` to `new_state`. It must take
      two arguments and return a nested structures of tensors. The structure of
      `new_state` must match the structure of `initial_state`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    use_inter_op_parallelism: An optional `bool`. Defaults to `True`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `output_types`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ReduceDataset', name, input_dataset, initial_state, other_arguments, 'f', f, 'output_types', output_types, 'output_shapes', output_shapes, 'use_inter_op_parallelism', use_inter_op_parallelism, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return reduce_dataset_eager_fallback(input_dataset, initial_state, other_arguments, f=f, output_types=output_types, output_shapes=output_shapes, use_inter_op_parallelism=use_inter_op_parallelism, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'reduce_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'reduce_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if use_inter_op_parallelism is None:
        use_inter_op_parallelism = True
    use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, 'use_inter_op_parallelism')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ReduceDataset', input_dataset=input_dataset, initial_state=initial_state, other_arguments=other_arguments, f=f, output_types=output_types, output_shapes=output_shapes, use_inter_op_parallelism=use_inter_op_parallelism, metadata=metadata, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('f', _op.get_attr('f'), 'Tstate', _op.get_attr('Tstate'), 'Targuments', _op.get_attr('Targuments'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'use_inter_op_parallelism', _op._get_attr_bool('use_inter_op_parallelism'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ReduceDataset', _inputs_flat, _attrs, _result)
    return _result