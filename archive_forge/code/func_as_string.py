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
@tf_export('strings.as_string', 'as_string', v1=['dtypes.as_string', 'strings.as_string', 'as_string'])
@deprecated_endpoints('dtypes.as_string')
def as_string(input: _atypes.TensorFuzzingAnnotation[TV_AsString_T], precision: int=-1, scientific: bool=False, shortest: bool=False, width: int=-1, fill: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Converts each entry in the given tensor to strings.

  Supports many numeric types and boolean.

  For Unicode, see the
  [https://www.tensorflow.org/tutorials/representation/unicode](Working with Unicode text)
  tutorial.

  Examples:

  >>> tf.strings.as_string([3, 2])
  <tf.Tensor: shape=(2,), dtype=string, numpy=array([b'3', b'2'], dtype=object)>
  >>> tf.strings.as_string([3.1415926, 2.71828], precision=2).numpy()
  array([b'3.14', b'2.72'], dtype=object)

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`, `complex64`, `complex128`, `bool`, `variant`, `string`.
    precision: An optional `int`. Defaults to `-1`.
      The post-decimal precision to use for floating point numbers.
      Only used if precision > -1.
    scientific: An optional `bool`. Defaults to `False`.
      Use scientific notation for floating point numbers.
    shortest: An optional `bool`. Defaults to `False`.
      Use shortest representation (either scientific or standard) for
      floating point numbers.
    width: An optional `int`. Defaults to `-1`.
      Pad pre-decimal numbers to this width.
      Applies to both floating point and integer numbers.
      Only used if width > -1.
    fill: An optional `string`. Defaults to `""`.
      The value to pad if width > -1.  If empty, pads with spaces.
      Another typical value is '0'.  String cannot be longer than 1 character.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'AsString', name, input, 'precision', precision, 'scientific', scientific, 'shortest', shortest, 'width', width, 'fill', fill)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_as_string((input, precision, scientific, shortest, width, fill, name), None)
            if _result is not NotImplemented:
                return _result
            return as_string_eager_fallback(input, precision=precision, scientific=scientific, shortest=shortest, width=width, fill=fill, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(as_string, (), dict(input=input, precision=precision, scientific=scientific, shortest=shortest, width=width, fill=fill, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_as_string((input, precision, scientific, shortest, width, fill, name), None)
        if _result is not NotImplemented:
            return _result
    if precision is None:
        precision = -1
    precision = _execute.make_int(precision, 'precision')
    if scientific is None:
        scientific = False
    scientific = _execute.make_bool(scientific, 'scientific')
    if shortest is None:
        shortest = False
    shortest = _execute.make_bool(shortest, 'shortest')
    if width is None:
        width = -1
    width = _execute.make_int(width, 'width')
    if fill is None:
        fill = ''
    fill = _execute.make_str(fill, 'fill')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('AsString', input=input, precision=precision, scientific=scientific, shortest=shortest, width=width, fill=fill, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(as_string, (), dict(input=input, precision=precision, scientific=scientific, shortest=shortest, width=width, fill=fill, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'precision', _op._get_attr_int('precision'), 'scientific', _op._get_attr_bool('scientific'), 'shortest', _op._get_attr_bool('shortest'), 'width', _op._get_attr_int('width'), 'fill', _op.get_attr('fill'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('AsString', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result