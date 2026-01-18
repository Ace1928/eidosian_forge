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
def decode_jpeg(contents: _atypes.TensorFuzzingAnnotation[_atypes.String], channels: int=0, ratio: int=1, fancy_upscaling: bool=True, try_recover_truncated: bool=False, acceptable_fraction: float=1, dct_method: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.UInt8]:
    """Decode a JPEG-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the JPEG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.

  If needed, the JPEG-encoded image is transformed to match the requested number
  of color channels.

  The attr `ratio` allows downscaling the image by an integer factor during
  decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
  downscaling the image later.


  This op also supports decoding PNGs and non-animated GIFs since the interface is
  the same, though it is cleaner to use `tf.io.decode_image`.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The JPEG-encoded image.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    ratio: An optional `int`. Defaults to `1`. Downscaling ratio.
    fancy_upscaling: An optional `bool`. Defaults to `True`.
      If true use a slower but nicer upscaling of the
      chroma planes (yuv420/422 only).
    try_recover_truncated: An optional `bool`. Defaults to `False`.
      If true try to recover an image from truncated input.
    acceptable_fraction: An optional `float`. Defaults to `1`.
      The minimum required fraction of lines before a truncated
      input is accepted.
    dct_method: An optional `string`. Defaults to `""`.
      string specifying a hint about the algorithm used for
      decompression.  Defaults to "" which maps to a system-specific
      default.  Currently valid values are ["INTEGER_FAST",
      "INTEGER_ACCURATE"].  The hint may be ignored (e.g., the internal
      jpeg library changes to a version that does not have that specific
      option.)
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DecodeJpeg', name, contents, 'channels', channels, 'ratio', ratio, 'fancy_upscaling', fancy_upscaling, 'try_recover_truncated', try_recover_truncated, 'acceptable_fraction', acceptable_fraction, 'dct_method', dct_method)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return decode_jpeg_eager_fallback(contents, channels=channels, ratio=ratio, fancy_upscaling=fancy_upscaling, try_recover_truncated=try_recover_truncated, acceptable_fraction=acceptable_fraction, dct_method=dct_method, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if channels is None:
        channels = 0
    channels = _execute.make_int(channels, 'channels')
    if ratio is None:
        ratio = 1
    ratio = _execute.make_int(ratio, 'ratio')
    if fancy_upscaling is None:
        fancy_upscaling = True
    fancy_upscaling = _execute.make_bool(fancy_upscaling, 'fancy_upscaling')
    if try_recover_truncated is None:
        try_recover_truncated = False
    try_recover_truncated = _execute.make_bool(try_recover_truncated, 'try_recover_truncated')
    if acceptable_fraction is None:
        acceptable_fraction = 1
    acceptable_fraction = _execute.make_float(acceptable_fraction, 'acceptable_fraction')
    if dct_method is None:
        dct_method = ''
    dct_method = _execute.make_str(dct_method, 'dct_method')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DecodeJpeg', contents=contents, channels=channels, ratio=ratio, fancy_upscaling=fancy_upscaling, try_recover_truncated=try_recover_truncated, acceptable_fraction=acceptable_fraction, dct_method=dct_method, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('channels', _op._get_attr_int('channels'), 'ratio', _op._get_attr_int('ratio'), 'fancy_upscaling', _op._get_attr_bool('fancy_upscaling'), 'try_recover_truncated', _op._get_attr_bool('try_recover_truncated'), 'acceptable_fraction', _op.get_attr('acceptable_fraction'), 'dct_method', _op.get_attr('dct_method'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DecodeJpeg', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result