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
def encode_jpeg_variable_quality(images: _atypes.TensorFuzzingAnnotation[_atypes.UInt8], quality: _atypes.TensorFuzzingAnnotation[_atypes.Int32], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """JPEG encode input image with provided compression quality.

  `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.
  `quality` is an int32 jpeg compression quality value between 0 and 100.

  Args:
    images: A `Tensor` of type `uint8`. Images to adjust.  At least 3-D.
    quality: A `Tensor` of type `int32`. An int quality to encode to.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'EncodeJpegVariableQuality', name, images, quality)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return encode_jpeg_variable_quality_eager_fallback(images, quality, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('EncodeJpegVariableQuality', images=images, quality=quality, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ()
        _inputs_flat = _op.inputs
        _execute.record_gradient('EncodeJpegVariableQuality', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result