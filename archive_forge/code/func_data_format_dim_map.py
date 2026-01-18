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
def data_format_dim_map(x: _atypes.TensorFuzzingAnnotation[TV_DataFormatDimMap_T], src_format: str='NHWC', dst_format: str='NCHW', name=None) -> _atypes.TensorFuzzingAnnotation[TV_DataFormatDimMap_T]:
    """Returns the dimension index in the destination data format given the one in

  the source data format.

  Args:
    x: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      A Tensor with each element as a dimension index in source data format.
      Must be in the range [-4, 4).
    src_format: An optional `string`. Defaults to `"NHWC"`.
      source data format.
    dst_format: An optional `string`. Defaults to `"NCHW"`.
      destination data format.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `x`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DataFormatDimMap', name, x, 'src_format', src_format, 'dst_format', dst_format)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return data_format_dim_map_eager_fallback(x, src_format=src_format, dst_format=dst_format, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if src_format is None:
        src_format = 'NHWC'
    src_format = _execute.make_str(src_format, 'src_format')
    if dst_format is None:
        dst_format = 'NCHW'
    dst_format = _execute.make_str(dst_format, 'dst_format')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DataFormatDimMap', x=x, src_format=src_format, dst_format=dst_format, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'src_format', _op.get_attr('src_format'), 'dst_format', _op.get_attr('dst_format'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DataFormatDimMap', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result