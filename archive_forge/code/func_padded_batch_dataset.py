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
def padded_batch_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], batch_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], padded_shapes: List[_atypes.TensorFuzzingAnnotation[_atypes.Int64]], padding_values, output_shapes, metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that batches and pads `batch_size` elements from the input.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    batch_size: A `Tensor` of type `int64`.
      A scalar representing the number of elements to accumulate in a
      batch.
    padded_shapes: A list of at least 1 `Tensor` objects with type `int64`.
      A list of int64 tensors representing the desired padded shapes
      of the corresponding output components. These shapes may be partially
      specified, using `-1` to indicate that a particular dimension should be
      padded to the maximum size of all batch elements.
    padding_values: A list of `Tensor` objects.
      A list of scalars containing the padding value to use for
      each of the outputs.
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
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'PaddedBatchDataset', name, input_dataset, batch_size, padded_shapes, padding_values, 'output_shapes', output_shapes, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return padded_batch_dataset_eager_fallback(input_dataset, batch_size, padded_shapes, padding_values, output_shapes=output_shapes, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(padded_shapes, (list, tuple)):
        raise TypeError("Expected list for 'padded_shapes' argument to 'padded_batch_dataset' Op, not %r." % padded_shapes)
    _attr_N = len(padded_shapes)
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'padded_batch_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('PaddedBatchDataset', input_dataset=input_dataset, batch_size=batch_size, padded_shapes=padded_shapes, padding_values=padding_values, output_shapes=output_shapes, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Toutput_types', _op.get_attr('Toutput_types'), 'output_shapes', _op.get_attr('output_shapes'), 'N', _op._get_attr_int('N'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('PaddedBatchDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result