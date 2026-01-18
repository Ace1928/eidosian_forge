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
def batch_dataset_v2(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], batch_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], drop_remainder: _atypes.TensorFuzzingAnnotation[_atypes.Bool], output_types, output_shapes, parallel_copy: bool=False, metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that batches `batch_size` elements from `input_dataset`.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a batch.
    drop_remainder: A `Tensor` of type `bool`.
      A scalar representing whether the last batch should be dropped in case its size
      is smaller than desired.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    parallel_copy: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BatchDatasetV2', name, input_dataset, batch_size, drop_remainder, 'parallel_copy', parallel_copy, 'output_types', output_types, 'output_shapes', output_shapes, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return batch_dataset_v2_eager_fallback(input_dataset, batch_size, drop_remainder, parallel_copy=parallel_copy, output_types=output_types, output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'batch_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'batch_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if parallel_copy is None:
        parallel_copy = False
    parallel_copy = _execute.make_bool(parallel_copy, 'parallel_copy')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BatchDatasetV2', input_dataset=input_dataset, batch_size=batch_size, drop_remainder=drop_remainder, output_types=output_types, output_shapes=output_shapes, parallel_copy=parallel_copy, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('parallel_copy', _op._get_attr_bool('parallel_copy'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('BatchDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result