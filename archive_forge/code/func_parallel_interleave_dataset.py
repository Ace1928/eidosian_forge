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
def parallel_interleave_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], other_arguments, cycle_length: _atypes.TensorFuzzingAnnotation[_atypes.Int64], block_length: _atypes.TensorFuzzingAnnotation[_atypes.Int64], sloppy: _atypes.TensorFuzzingAnnotation[_atypes.Bool], buffer_output_elements: _atypes.TensorFuzzingAnnotation[_atypes.Int64], prefetch_input_elements: _atypes.TensorFuzzingAnnotation[_atypes.Int64], f, output_types, output_shapes, metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that applies `f` to the outputs of `input_dataset`.

  The resulting dataset is similar to the `InterleaveDataset`, with the exception
  that if retrieving the next value from a dataset would cause the requester to
  block, it will skip that input dataset. This dataset is especially useful
  when loading data from a variable-latency datastores (e.g. HDFS, GCS), as it
  allows the training step to proceed so long as some data is available.

  !! WARNING !! If the `sloppy` parameter is set to `True`, the operation of this
  dataset will not be deterministic!

  This dataset has been superseded by `ParallelInterleaveDatasetV2`.  New code
  should use `ParallelInterleaveDatasetV2`.

  The Python API `tf.data.experimental.parallel_interleave` creates instances of
  this op. `tf.data.experimental.parallel_interleave` is a deprecated API.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      Dataset that produces a stream of arguments for the function `f`.
    other_arguments: A list of `Tensor` objects.
      Additional arguments to pass to `f` beyond those produced by `input_dataset`.
      Evaluated once when the dataset is instantiated.
    cycle_length: A `Tensor` of type `int64`.
      Number of datasets (each created by applying `f` to the elements of
      `input_dataset`) among which the `ParallelInterleaveDataset` will cycle in a
      round-robin fashion.
    block_length: A `Tensor` of type `int64`.
      Number of elements at a time to produce from each interleaved invocation of a
      dataset returned by `f`.
    sloppy: A `Tensor` of type `bool`.
      If `True`, return elements as they become available, even if that means returning
      these elements in a non-deterministic order. Sloppy operation may result in better
      performance in the presence of stragglers, but the dataset will still block if
      all of its open streams are blocked.
      If `False`, always return elements in a deterministic order.
    buffer_output_elements: A `Tensor` of type `int64`.
      The number of elements each iterator being interleaved should buffer (similar
      to the `.prefetch()` transformation for each interleaved iterator).
    prefetch_input_elements: A `Tensor` of type `int64`.
      Determines the number of iterators to prefetch, allowing buffers to warm up and
      data to be pre-fetched without blocking the main thread.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
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
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ParallelInterleaveDataset', name, input_dataset, other_arguments, cycle_length, block_length, sloppy, buffer_output_elements, prefetch_input_elements, 'f', f, 'output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return parallel_interleave_dataset_eager_fallback(input_dataset, other_arguments, cycle_length, block_length, sloppy, buffer_output_elements, prefetch_input_elements, f=f, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'parallel_interleave_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'parallel_interleave_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ParallelInterleaveDataset', input_dataset=input_dataset, other_arguments=other_arguments, cycle_length=cycle_length, block_length=block_length, sloppy=sloppy, buffer_output_elements=buffer_output_elements, prefetch_input_elements=prefetch_input_elements, f=f, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('f', _op.get_attr('f'), 'Targuments', _op.get_attr('Targuments'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ParallelInterleaveDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result