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
def parallel_map_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], other_arguments, num_parallel_calls: _atypes.TensorFuzzingAnnotation[_atypes.Int32], f, output_types, output_shapes, use_inter_op_parallelism: bool=True, sloppy: bool=False, preserve_cardinality: bool=False, metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that applies `f` to the outputs of `input_dataset`.

  Unlike a "MapDataset", which applies `f` sequentially, this dataset invokes up
  to `num_parallel_calls` copies of `f` in parallel.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    other_arguments: A list of `Tensor` objects.
    num_parallel_calls: A `Tensor` of type `int32`.
      The number of concurrent invocations of `f` that process
      elements from `input_dataset` in parallel.
    f: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    use_inter_op_parallelism: An optional `bool`. Defaults to `True`.
    sloppy: An optional `bool`. Defaults to `False`.
    preserve_cardinality: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ParallelMapDataset', name, input_dataset, other_arguments, num_parallel_calls, 'f', f, 'output_types', output_types, 'output_shapes', output_shapes, 'use_inter_op_parallelism', use_inter_op_parallelism, 'sloppy', sloppy, 'preserve_cardinality', preserve_cardinality, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return parallel_map_dataset_eager_fallback(input_dataset, other_arguments, num_parallel_calls, f=f, output_types=output_types, output_shapes=output_shapes, use_inter_op_parallelism=use_inter_op_parallelism, sloppy=sloppy, preserve_cardinality=preserve_cardinality, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'parallel_map_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'parallel_map_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if use_inter_op_parallelism is None:
        use_inter_op_parallelism = True
    use_inter_op_parallelism = _execute.make_bool(use_inter_op_parallelism, 'use_inter_op_parallelism')
    if sloppy is None:
        sloppy = False
    sloppy = _execute.make_bool(sloppy, 'sloppy')
    if preserve_cardinality is None:
        preserve_cardinality = False
    preserve_cardinality = _execute.make_bool(preserve_cardinality, 'preserve_cardinality')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ParallelMapDataset', input_dataset=input_dataset, other_arguments=other_arguments, num_parallel_calls=num_parallel_calls, f=f, output_types=output_types, output_shapes=output_shapes, use_inter_op_parallelism=use_inter_op_parallelism, sloppy=sloppy, preserve_cardinality=preserve_cardinality, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('f', _op.get_attr('f'), 'Targuments', _op.get_attr('Targuments'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'use_inter_op_parallelism', _op._get_attr_bool('use_inter_op_parallelism'), 'sloppy', _op._get_attr_bool('sloppy'), 'preserve_cardinality', _op._get_attr_bool('preserve_cardinality'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ParallelMapDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result