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
def substr(input: _atypes.TensorFuzzingAnnotation[_atypes.String], pos: _atypes.TensorFuzzingAnnotation[TV_Substr_T], len: _atypes.TensorFuzzingAnnotation[TV_Substr_T], unit: str='BYTE', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Return substrings from `Tensor` of strings.

  For each string in the input `Tensor`, creates a substring starting at index
  `pos` with a total length of `len`.

  If `len` defines a substring that would extend beyond the length of the input
  string, or if `len` is negative, then as many characters as possible are used.

  A negative `pos` indicates distance within the string backwards from the end.

  If `pos` specifies an index which is out of range for any of the input strings,
  then an `InvalidArgumentError` is thrown.

  `pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
  Op creation.

  *NOTE*: `Substr` supports broadcasting up to two dimensions. More about
  broadcasting
  [here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

  ---

  Examples

  Using scalar `pos` and `len`:

  ```python
  input = [b'Hello', b'World']
  position = 1
  length = 3

  output = [b'ell', b'orl']
  ```

  Using `pos` and `len` with same shape as `input`:

  ```python
  input = [[b'ten', b'eleven', b'twelve'],
           [b'thirteen', b'fourteen', b'fifteen'],
           [b'sixteen', b'seventeen', b'eighteen']]
  position = [[1, 2, 3],
              [1, 2, 3],
              [1, 2, 3]]
  length =   [[2, 3, 4],
              [4, 3, 2],
              [5, 5, 5]]

  output = [[b'en', b'eve', b'lve'],
            [b'hirt', b'urt', b'te'],
            [b'ixtee', b'vente', b'hteen']]
  ```

  Broadcasting `pos` and `len` onto `input`:

  ```
  input = [[b'ten', b'eleven', b'twelve'],
           [b'thirteen', b'fourteen', b'fifteen'],
           [b'sixteen', b'seventeen', b'eighteen'],
           [b'nineteen', b'twenty', b'twentyone']]
  position = [1, 2, 3]
  length =   [1, 2, 3]

  output = [[b'e', b'ev', b'lve'],
            [b'h', b'ur', b'tee'],
            [b'i', b've', b'hte'],
            [b'i', b'en', b'nty']]
  ```

  Broadcasting `input` onto `pos` and `len`:

  ```
  input = b'thirteen'
  position = [1, 5, 7]
  length =   [3, 2, 1]

  output = [b'hir', b'ee', b'n']
  ```

  Raises:

    * `ValueError`: If the first argument cannot be converted to a
       Tensor of `dtype string`.
    * `InvalidArgumentError`: If indices are out of range.
    * `ValueError`: If `pos` and `len` are not the same shape.

  Args:
    input: A `Tensor` of type `string`. Tensor of strings
    pos: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      Scalar defining the position of first character in each substring
    len: A `Tensor`. Must have the same type as `pos`.
      Scalar defining the number of characters to include in each substring
    unit: An optional `string` from: `"BYTE", "UTF8_CHAR"`. Defaults to `"BYTE"`.
      The unit that is used to create the substring.  One of: `"BYTE"` (for
      defining position and length by bytes) or `"UTF8_CHAR"` (for the UTF-8
      encoded Unicode code points).  The default is `"BYTE"`. Results are undefined if
      `unit=UTF8_CHAR` and the `input` strings do not contain structurally valid
      UTF-8.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Substr', name, input, pos, len, 'unit', unit)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return substr_eager_fallback(input, pos, len, unit=unit, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if unit is None:
        unit = 'BYTE'
    unit = _execute.make_str(unit, 'unit')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Substr', input=input, pos=pos, len=len, unit=unit, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'unit', _op.get_attr('unit'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Substr', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result