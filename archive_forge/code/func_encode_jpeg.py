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
def encode_jpeg(image: _atypes.TensorFuzzingAnnotation[_atypes.UInt8], format: str='', quality: int=95, progressive: bool=False, optimize_size: bool=False, chroma_downsampling: bool=True, density_unit: str='in', x_density: int=300, y_density: int=300, xmp_metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """JPEG-encode an image.

  `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

  The attr `format` can be used to override the color format of the encoded
  output.  Values can be:

  *   `''`: Use a default format based on the number of channels in the image.
  *   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
      of `image` must be 1.
  *   `rgb`: Output an RGB JPEG image. The `channels` dimension
      of `image` must be 3.

  If `format` is not specified or is the empty string, a default format is picked
  in function of the number of channels in `image`:

  *   1: Output a grayscale image.
  *   3: Output an RGB image.

  Args:
    image: A `Tensor` of type `uint8`.
      3-D with shape `[height, width, channels]`.
    format: An optional `string` from: `"", "grayscale", "rgb"`. Defaults to `""`.
      Per pixel image format.
    quality: An optional `int`. Defaults to `95`.
      Quality of the compression from 0 to 100 (higher is better and slower).
    progressive: An optional `bool`. Defaults to `False`.
      If True, create a JPEG that loads progressively (coarse to fine).
    optimize_size: An optional `bool`. Defaults to `False`.
      If True, spend CPU/RAM to reduce size with no quality change.
    chroma_downsampling: An optional `bool`. Defaults to `True`.
      See http://en.wikipedia.org/wiki/Chroma_subsampling.
    density_unit: An optional `string` from: `"in", "cm"`. Defaults to `"in"`.
      Unit used to specify `x_density` and `y_density`:
      pixels per inch (`'in'`) or centimeter (`'cm'`).
    x_density: An optional `int`. Defaults to `300`.
      Horizontal pixels per density unit.
    y_density: An optional `int`. Defaults to `300`.
      Vertical pixels per density unit.
    xmp_metadata: An optional `string`. Defaults to `""`.
      If not empty, embed this XMP metadata in the image header.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'EncodeJpeg', name, image, 'format', format, 'quality', quality, 'progressive', progressive, 'optimize_size', optimize_size, 'chroma_downsampling', chroma_downsampling, 'density_unit', density_unit, 'x_density', x_density, 'y_density', y_density, 'xmp_metadata', xmp_metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return encode_jpeg_eager_fallback(image, format=format, quality=quality, progressive=progressive, optimize_size=optimize_size, chroma_downsampling=chroma_downsampling, density_unit=density_unit, x_density=x_density, y_density=y_density, xmp_metadata=xmp_metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if format is None:
        format = ''
    format = _execute.make_str(format, 'format')
    if quality is None:
        quality = 95
    quality = _execute.make_int(quality, 'quality')
    if progressive is None:
        progressive = False
    progressive = _execute.make_bool(progressive, 'progressive')
    if optimize_size is None:
        optimize_size = False
    optimize_size = _execute.make_bool(optimize_size, 'optimize_size')
    if chroma_downsampling is None:
        chroma_downsampling = True
    chroma_downsampling = _execute.make_bool(chroma_downsampling, 'chroma_downsampling')
    if density_unit is None:
        density_unit = 'in'
    density_unit = _execute.make_str(density_unit, 'density_unit')
    if x_density is None:
        x_density = 300
    x_density = _execute.make_int(x_density, 'x_density')
    if y_density is None:
        y_density = 300
    y_density = _execute.make_int(y_density, 'y_density')
    if xmp_metadata is None:
        xmp_metadata = ''
    xmp_metadata = _execute.make_str(xmp_metadata, 'xmp_metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('EncodeJpeg', image=image, format=format, quality=quality, progressive=progressive, optimize_size=optimize_size, chroma_downsampling=chroma_downsampling, density_unit=density_unit, x_density=x_density, y_density=y_density, xmp_metadata=xmp_metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('format', _op.get_attr('format'), 'quality', _op._get_attr_int('quality'), 'progressive', _op._get_attr_bool('progressive'), 'optimize_size', _op._get_attr_bool('optimize_size'), 'chroma_downsampling', _op._get_attr_bool('chroma_downsampling'), 'density_unit', _op.get_attr('density_unit'), 'x_density', _op._get_attr_int('x_density'), 'y_density', _op._get_attr_int('y_density'), 'xmp_metadata', _op.get_attr('xmp_metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('EncodeJpeg', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result