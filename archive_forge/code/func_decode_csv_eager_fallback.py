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
def decode_csv_eager_fallback(records: _atypes.TensorFuzzingAnnotation[_atypes.String], record_defaults, field_delim: str, use_quote_delim: bool, na_value: str, select_cols, name, ctx):
    if field_delim is None:
        field_delim = ','
    field_delim = _execute.make_str(field_delim, 'field_delim')
    if use_quote_delim is None:
        use_quote_delim = True
    use_quote_delim = _execute.make_bool(use_quote_delim, 'use_quote_delim')
    if na_value is None:
        na_value = ''
    na_value = _execute.make_str(na_value, 'na_value')
    if select_cols is None:
        select_cols = []
    if not isinstance(select_cols, (list, tuple)):
        raise TypeError("Expected list for 'select_cols' argument to 'decode_csv' Op, not %r." % select_cols)
    select_cols = [_execute.make_int(_i, 'select_cols') for _i in select_cols]
    _attr_OUT_TYPE, record_defaults = _execute.convert_to_mixed_eager_tensors(record_defaults, ctx)
    records = _ops.convert_to_tensor(records, _dtypes.string)
    _inputs_flat = [records] + list(record_defaults)
    _attrs = ('OUT_TYPE', _attr_OUT_TYPE, 'field_delim', field_delim, 'use_quote_delim', use_quote_delim, 'na_value', na_value, 'select_cols', select_cols)
    _result = _execute.execute(b'DecodeCSV', len(record_defaults), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DecodeCSV', _inputs_flat, _attrs, _result)
    return _result