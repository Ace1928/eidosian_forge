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
def infeed_enqueue(input: _atypes.TensorFuzzingAnnotation[TV_InfeedEnqueue_dtype], shape=[], layout=[], device_ordinal: int=-1, name=None):
    """An op which feeds a single Tensor value into the computation.

  Args:
    input: A `Tensor`.
      A tensor that will be provided using the infeed mechanism.
    shape: An optional `tf.TensorShape` or list of `ints`. Defaults to `[]`.
      The shape of the tensor.
    layout: An optional list of `ints`. Defaults to `[]`.
      A vector holding the requested layout in minor-to-major sequence.
      If a layout attribute is passed, but its values are all -1, the layout will
      be computed by the infeed operation.
    device_ordinal: An optional `int`. Defaults to `-1`.
      The TPU device to use. This should be -1 when the Op
      is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'InfeedEnqueue', name, input, 'shape', shape, 'layout', layout, 'device_ordinal', device_ordinal)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return infeed_enqueue_eager_fallback(input, shape=shape, layout=layout, device_ordinal=device_ordinal, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if shape is None:
        shape = []
    shape = _execute.make_shape(shape, 'shape')
    if layout is None:
        layout = []
    if not isinstance(layout, (list, tuple)):
        raise TypeError("Expected list for 'layout' argument to 'infeed_enqueue' Op, not %r." % layout)
    layout = [_execute.make_int(_i, 'layout') for _i in layout]
    if device_ordinal is None:
        device_ordinal = -1
    device_ordinal = _execute.make_int(device_ordinal, 'device_ordinal')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('InfeedEnqueue', input=input, shape=shape, layout=layout, device_ordinal=device_ordinal, name=name)
    return _op