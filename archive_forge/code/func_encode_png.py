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
def encode_png(image: _atypes.TensorFuzzingAnnotation[TV_EncodePng_T], compression: int=-1, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """PNG-encode an image.

  `image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
  where `channels` is:

  *   1: for grayscale.
  *   2: for grayscale + alpha.
  *   3: for RGB.
  *   4: for RGBA.

  The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
  default or a value from 0 to 9.  9 is the highest compression level, generating
  the smallest output, but is slower.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`.
      3-D with shape `[height, width, channels]`.
    compression: An optional `int`. Defaults to `-1`. Compression level.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'EncodePng', name, image, 'compression', compression)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return encode_png_eager_fallback(image, compression=compression, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if compression is None:
        compression = -1
    compression = _execute.make_int(compression, 'compression')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('EncodePng', image=image, compression=compression, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('compression', _op._get_attr_int('compression'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('EncodePng', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result