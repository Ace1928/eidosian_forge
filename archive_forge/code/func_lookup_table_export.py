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
def lookup_table_export(table_handle: _atypes.TensorFuzzingAnnotation[_atypes.String], Tkeys: TV_LookupTableExport_Tkeys, Tvalues: TV_LookupTableExport_Tvalues, name=None):
    """Outputs all keys and values in the table.

  Args:
    table_handle: A `Tensor` of type mutable `string`. Handle to the table.
    Tkeys: A `tf.DType`.
    Tvalues: A `tf.DType`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (keys, values).

    keys: A `Tensor` of type `Tkeys`.
    values: A `Tensor` of type `Tvalues`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        raise RuntimeError("lookup_table_export op does not support eager execution. Arg 'table_handle' is a ref.")
    Tkeys = _execute.make_type(Tkeys, 'Tkeys')
    Tvalues = _execute.make_type(Tvalues, 'Tvalues')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('LookupTableExport', table_handle=table_handle, Tkeys=Tkeys, Tvalues=Tvalues, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tkeys', _op._get_attr_type('Tkeys'), 'Tvalues', _op._get_attr_type('Tvalues'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('LookupTableExport', _inputs_flat, _attrs, _result)
    _result = _LookupTableExportOutput._make(_result)
    return _result