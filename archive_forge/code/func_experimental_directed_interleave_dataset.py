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
def experimental_directed_interleave_dataset(selector_input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], data_input_datasets: List[_atypes.TensorFuzzingAnnotation[_atypes.Variant]], output_types, output_shapes, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """A substitute for `InterleaveDataset` on a fixed list of `N` datasets.

  Args:
    selector_input_dataset: A `Tensor` of type `variant`.
      A dataset of scalar `DT_INT64` elements that determines which of the
      `N` data inputs should produce the next output element.
    data_input_datasets: A list of at least 1 `Tensor` objects with type `variant`.
      `N` datasets with the same type that will be interleaved according to
      the values of `selector_input_dataset`.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ExperimentalDirectedInterleaveDataset', name, selector_input_dataset, data_input_datasets, 'output_types', output_types, 'output_shapes', output_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return experimental_directed_interleave_dataset_eager_fallback(selector_input_dataset, data_input_datasets, output_types=output_types, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(data_input_datasets, (list, tuple)):
        raise TypeError("Expected list for 'data_input_datasets' argument to 'experimental_directed_interleave_dataset' Op, not %r." % data_input_datasets)
    _attr_N = len(data_input_datasets)
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'experimental_directed_interleave_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'experimental_directed_interleave_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ExperimentalDirectedInterleaveDataset', selector_input_dataset=selector_input_dataset, data_input_datasets=data_input_datasets, output_types=output_types, output_shapes=output_shapes, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'N', _op._get_attr_int('N'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ExperimentalDirectedInterleaveDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result