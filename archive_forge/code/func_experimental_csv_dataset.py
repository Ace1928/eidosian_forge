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
def experimental_csv_dataset(filenames: _atypes.TensorFuzzingAnnotation[_atypes.String], compression_type: _atypes.TensorFuzzingAnnotation[_atypes.String], buffer_size: _atypes.TensorFuzzingAnnotation[_atypes.Int64], header: _atypes.TensorFuzzingAnnotation[_atypes.Bool], field_delim: _atypes.TensorFuzzingAnnotation[_atypes.String], use_quote_delim: _atypes.TensorFuzzingAnnotation[_atypes.Bool], na_value: _atypes.TensorFuzzingAnnotation[_atypes.String], select_cols: _atypes.TensorFuzzingAnnotation[_atypes.Int64], record_defaults, output_shapes, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """TODO: add doc.

  Args:
    filenames: A `Tensor` of type `string`.
    compression_type: A `Tensor` of type `string`.
    buffer_size: A `Tensor` of type `int64`.
    header: A `Tensor` of type `bool`.
    field_delim: A `Tensor` of type `string`.
    use_quote_delim: A `Tensor` of type `bool`.
    na_value: A `Tensor` of type `string`.
    select_cols: A `Tensor` of type `int64`.
    record_defaults: A list of `Tensor` objects with types from: `float32`, `float64`, `int32`, `int64`, `string`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ExperimentalCSVDataset', name, filenames, compression_type, buffer_size, header, field_delim, use_quote_delim, na_value, select_cols, record_defaults, 'output_shapes', output_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return experimental_csv_dataset_eager_fallback(filenames, compression_type, buffer_size, header, field_delim, use_quote_delim, na_value, select_cols, record_defaults, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'experimental_csv_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ExperimentalCSVDataset', filenames=filenames, compression_type=compression_type, buffer_size=buffer_size, header=header, field_delim=field_delim, use_quote_delim=use_quote_delim, na_value=na_value, select_cols=select_cols, record_defaults=record_defaults, output_shapes=output_shapes, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ExperimentalCSVDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result