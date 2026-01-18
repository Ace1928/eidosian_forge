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
def parallel_interleave_dataset_v3(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], other_arguments, cycle_length: _atypes.TensorFuzzingAnnotation[_atypes.Int64], block_length: _atypes.TensorFuzzingAnnotation[_atypes.Int64], num_parallel_calls: _atypes.TensorFuzzingAnnotation[_atypes.Int64], f, output_types, output_shapes, deterministic: str='default', metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that applies `f` to the outputs of `input_dataset`.

  The resulting dataset is similar to the `InterleaveDataset`, except that the
  dataset will fetch records from the interleaved datasets in parallel.

  The `tf.data` Python API creates instances of this op from
  `Dataset.interleave()` when the `num_parallel_calls` parameter of that method
  is set to any value other than `None`.

  By default, the output of this dataset will be deterministic, which may result
  in the dataset blocking if the next data item to be returned isn't available.
  In order to avoid head-of-line blocking, one can either set the `deterministic`
  attribute to "false", or leave it as "default" and set the
  `experimental_deterministic` parameter of `tf.data.Options` to `False`.
  This can improve performance at the expense of non-determinism.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      Dataset that produces a stream of arguments for the function `f`.
    other_arguments: A list of `Tensor` objects.
      Additional arguments to pass to `f` beyond those produced by `input_dataset`.
      Evaluated once when the dataset is instantiated.
    cycle_length: A `Tensor` of type `int64`.
      Number of datasets (each created by applying `f` to the elements of
      `input_dataset`) among which the `ParallelInterleaveDatasetV2` will cycle in a
      round-robin fashion.
    block_length: A `Tensor` of type `int64`.
      Number of elements at a time to produce from each interleaved invocation of a
      dataset returned by `f`.
    num_parallel_calls: A `Tensor` of type `int64`.
      Determines the number of threads that should be used for fetching data from
      input datasets in parallel. The Python API `tf.data.experimental.AUTOTUNE`
      constant can be used to indicate that the level of parallelism should be autotuned.
    f: A function decorated with @Defun.
      A function mapping elements of `input_dataset`, concatenated with
      `other_arguments`, to a Dataset variant that contains elements matching
      `output_types` and `output_shapes`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    deterministic: An optional `string`. Defaults to `"default"`.
      A string indicating the op-level determinism to use. Deterministic controls
      whether the interleave is allowed to return elements out of order if the next
      element to be returned isn't available, but a later element is. Options are
      "true", "false", and "default". "default" indicates that determinism should be
      decided by the `experimental_deterministic` parameter of `tf.data.Options`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ParallelInterleaveDatasetV3', name, input_dataset, other_arguments, cycle_length, block_length, num_parallel_calls, 'f', f, 'deterministic', deterministic, 'output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return parallel_interleave_dataset_v3_eager_fallback(input_dataset, other_arguments, cycle_length, block_length, num_parallel_calls, f=f, deterministic=deterministic, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'parallel_interleave_dataset_v3' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'parallel_interleave_dataset_v3' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if deterministic is None:
        deterministic = 'default'
    deterministic = _execute.make_str(deterministic, 'deterministic')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ParallelInterleaveDatasetV3', input_dataset=input_dataset, other_arguments=other_arguments, cycle_length=cycle_length, block_length=block_length, num_parallel_calls=num_parallel_calls, f=f, output_types=output_types, output_shapes=output_shapes, deterministic=deterministic, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('f', _op.get_attr('f'), 'deterministic', _op.get_attr('deterministic'), 'Targuments', _op.get_attr('Targuments'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ParallelInterleaveDatasetV3', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result