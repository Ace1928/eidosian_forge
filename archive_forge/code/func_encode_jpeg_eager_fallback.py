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
def encode_jpeg_eager_fallback(image: _atypes.TensorFuzzingAnnotation[_atypes.UInt8], format: str, quality: int, progressive: bool, optimize_size: bool, chroma_downsampling: bool, density_unit: str, x_density: int, y_density: int, xmp_metadata: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
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
    image = _ops.convert_to_tensor(image, _dtypes.uint8)
    _inputs_flat = [image]
    _attrs = ('format', format, 'quality', quality, 'progressive', progressive, 'optimize_size', optimize_size, 'chroma_downsampling', chroma_downsampling, 'density_unit', density_unit, 'x_density', x_density, 'y_density', y_density, 'xmp_metadata', xmp_metadata)
    _result = _execute.execute(b'EncodeJpeg', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('EncodeJpeg', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result