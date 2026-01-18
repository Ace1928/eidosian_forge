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
def extract_glimpse(input: _atypes.TensorFuzzingAnnotation[_atypes.Float32], size: _atypes.TensorFuzzingAnnotation[_atypes.Int32], offsets: _atypes.TensorFuzzingAnnotation[_atypes.Float32], centered: bool=True, normalized: bool=True, uniform_noise: bool=True, noise: str='uniform', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float32]:
    """Extracts a glimpse from the input tensor.

  Returns a set of windows called glimpses extracted at location
  `offsets` from the input tensor. If the windows only partially
  overlaps the inputs, the non overlapping areas will be filled with
  random noise.

  The result is a 4-D tensor of shape `[batch_size, glimpse_height,
  glimpse_width, channels]`. The channels and batch dimensions are the
  same as that of the input tensor. The height and width of the output
  windows are specified in the `size` parameter.

  The argument `normalized` and `centered` controls how the windows are built:

  * If the coordinates are normalized but not centered, 0.0 and 1.0
    correspond to the minimum and maximum of each height and width
    dimension.
  * If the coordinates are both normalized and centered, they range from
    -1.0 to 1.0. The coordinates (-1.0, -1.0) correspond to the upper
    left corner, the lower right corner is located at (1.0, 1.0) and the
    center is at (0, 0).
  * If the coordinates are not normalized they are interpreted as
    numbers of pixels.

  Args:
    input: A `Tensor` of type `float32`.
      A 4-D float tensor of shape `[batch_size, height, width, channels]`.
    size: A `Tensor` of type `int32`.
      A 1-D tensor of 2 elements containing the size of the glimpses
      to extract.  The glimpse height must be specified first, following
      by the glimpse width.
    offsets: A `Tensor` of type `float32`.
      A 2-D integer tensor of shape `[batch_size, 2]` containing
      the y, x locations of the center of each window.
    centered: An optional `bool`. Defaults to `True`.
      indicates if the offset coordinates are centered relative to
      the image, in which case the (0, 0) offset is relative to the center
      of the input images. If false, the (0,0) offset corresponds to the
      upper left corner of the input images.
    normalized: An optional `bool`. Defaults to `True`.
      indicates if the offset coordinates are normalized.
    uniform_noise: An optional `bool`. Defaults to `True`.
      indicates if the noise should be generated using a
      uniform distribution or a Gaussian distribution.
    noise: An optional `string`. Defaults to `"uniform"`.
      indicates if the noise should `uniform`, `gaussian`, or
      `zero`. The default is `uniform` which means the noise type
      will be decided by `uniform_noise`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ExtractGlimpse', name, input, size, offsets, 'centered', centered, 'normalized', normalized, 'uniform_noise', uniform_noise, 'noise', noise)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return extract_glimpse_eager_fallback(input, size, offsets, centered=centered, normalized=normalized, uniform_noise=uniform_noise, noise=noise, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if centered is None:
        centered = True
    centered = _execute.make_bool(centered, 'centered')
    if normalized is None:
        normalized = True
    normalized = _execute.make_bool(normalized, 'normalized')
    if uniform_noise is None:
        uniform_noise = True
    uniform_noise = _execute.make_bool(uniform_noise, 'uniform_noise')
    if noise is None:
        noise = 'uniform'
    noise = _execute.make_str(noise, 'noise')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('ExtractGlimpse', input=input, size=size, offsets=offsets, centered=centered, normalized=normalized, uniform_noise=uniform_noise, noise=noise, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('centered', _op._get_attr_bool('centered'), 'normalized', _op._get_attr_bool('normalized'), 'uniform_noise', _op._get_attr_bool('uniform_noise'), 'noise', _op.get_attr('noise'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ExtractGlimpse', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result