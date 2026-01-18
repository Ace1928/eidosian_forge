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
def experimental_map_and_batch_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], other_arguments, batch_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_parallel_calls: _atypes.TensorFuzzingAnnotation[_atypes.Int64], drop_remainder: _atypes.TensorFuzzingAnnotation[_atypes.Bool], f, output_types, output_shapes, preserve_cardinality: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that fuses mapping with batching.

  Creates a dataset that applies `f` to the outputs of `input_dataset` and then
  batches `batch_size` of them.

  Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
  to `batch_size * num_parallel_batches` copies of `f` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    other_arguments: A list of `Tensor` objects.
      A list of tensors, typically values that were captured when building a closure
      for `f`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch. It determines the number of concurrent invocations of `f` that process
      elements from `input_dataset` in parallel.
    num_parallel_calls: A `Tensor` of type `int64`.
      A scalar representing the maximum number of parallel invocations of the `map_fn`
      function. Applying the `map_fn` on consecutive input elements in parallel has
      the potential to improve input pipeline throughput.
    drop_remainder: A `Tensor` of type `bool`.
      A scalar representing whether the last batch should be dropped in case its size
      is smaller than desired.
    f: A function decorated with @Defun.
      A function to apply to the outputs of `input_dataset`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ExperimentalMapAndBatchDataset', name, input_dataset, other_arguments, batch_size, num_parallel_calls, drop_remainder, 'f', f, 'output_types', output_types, 'output_shapes', output_shapes, 'preserve_cardinality', preserve_cardinality)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return experimental_map_and_batch_dataset_eager_fallback(input_dataset, other_arguments, batch_size, num_parallel_calls, drop_remainder, f=f, output_types=output_types, output_shapes=output_shapes, preserve_cardinality=preserve_cardinality, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'experimental_map_and_batch_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'experimental_map_and_batch_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if preserve_cardinality is None:
        preserve_cardinality = False
    preserve_cardinality = _execute.make_bool(preserve_cardinality, 'preserve_cardinality')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ExperimentalMapAndBatchDataset', input_dataset=input_dataset, other_arguments=other_arguments, batch_size=batch_size, num_parallel_calls=num_parallel_calls, drop_remainder=drop_remainder, f=f, output_types=output_types, output_shapes=output_shapes, preserve_cardinality=preserve_cardinality, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('f', _op.get_attr('f'), 'Targuments', _op.get_attr('Targuments'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'preserve_cardinality', _op._get_attr_bool('preserve_cardinality'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ExperimentalMapAndBatchDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result