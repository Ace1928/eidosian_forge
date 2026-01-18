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
def string_length(input: _atypes.TensorFuzzingAnnotation[_atypes.String], unit: str='BYTE', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    """String lengths of `input`.

  Computes the length of each string given in the input tensor.

  >>> strings = tf.constant(['Hello','TensorFlow', '\\U0001F642'])
  >>> tf.strings.length(strings).numpy() # default counts bytes
  array([ 5, 10, 4], dtype=int32)
  >>> tf.strings.length(strings, unit="UTF8_CHAR").numpy()
  array([ 5, 10, 1], dtype=int32)

  Args:
    input: A `Tensor` of type `string`.
      The strings for which to compute the length for each element.
    unit: An optional `string` from: `"BYTE", "UTF8_CHAR"`. Defaults to `"BYTE"`.
      The unit that is counted to compute string length.  One of: `"BYTE"` (for
      the number of bytes in each string) or `"UTF8_CHAR"` (for the number of UTF-8
      encoded Unicode code points in each string).  Results are undefined
      if `unit=UTF8_CHAR` and the `input` strings do not contain structurally
      valid UTF-8.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StringLength', name, input, 'unit', unit)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return string_length_eager_fallback(input, unit=unit, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if unit is None:
        unit = 'BYTE'
    unit = _execute.make_str(unit, 'unit')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StringLength', input=input, unit=unit, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('unit', _op.get_attr('unit'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StringLength', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result