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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('ragged_fill_empty_rows')
def ragged_fill_empty_rows(value_rowids: _atypes.TensorFuzzingAnnotation[_atypes.Int64], values: _atypes.TensorFuzzingAnnotation[TV_RaggedFillEmptyRows_T], nrows: _atypes.TensorFuzzingAnnotation[_atypes.Int64], default_value: _atypes.TensorFuzzingAnnotation[TV_RaggedFillEmptyRows_T], name=None):
    """TODO: add doc.

  Args:
    value_rowids: A `Tensor` of type `int64`.
    values: A `Tensor`.
    nrows: A `Tensor` of type `int64`.
    default_value: A `Tensor`. Must have the same type as `values`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_value_rowids, output_values, empty_row_indicator, reverse_index_map).

    output_value_rowids: A `Tensor` of type `int64`.
    output_values: A `Tensor`. Has the same type as `values`.
    empty_row_indicator: A `Tensor` of type `bool`.
    reverse_index_map: A `Tensor` of type `int64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RaggedFillEmptyRows', name, value_rowids, values, nrows, default_value)
            _result = _RaggedFillEmptyRowsOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_ragged_fill_empty_rows((value_rowids, values, nrows, default_value, name), None)
            if _result is not NotImplemented:
                return _result
            return ragged_fill_empty_rows_eager_fallback(value_rowids, values, nrows, default_value, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(ragged_fill_empty_rows, (), dict(value_rowids=value_rowids, values=values, nrows=nrows, default_value=default_value, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_ragged_fill_empty_rows((value_rowids, values, nrows, default_value, name), None)
        if _result is not NotImplemented:
            return _result
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('RaggedFillEmptyRows', value_rowids=value_rowids, values=values, nrows=nrows, default_value=default_value, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(ragged_fill_empty_rows, (), dict(value_rowids=value_rowids, values=values, nrows=nrows, default_value=default_value, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RaggedFillEmptyRows', _inputs_flat, _attrs, _result)
    _result = _RaggedFillEmptyRowsOutput._make(_result)
    return _result