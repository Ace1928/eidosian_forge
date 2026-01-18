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
def outfeed_dequeue_v2(device_ordinal: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtype: TV_OutfeedDequeueV2_dtype, shape, name=None) -> _atypes.TensorFuzzingAnnotation[TV_OutfeedDequeueV2_dtype]:
    """Retrieves a single tensor from the computation outfeed. Device ordinal is a
tensor allowing dynamic outfeed.

  This operation will block indefinitely until data is available.

  Args:
    device_ordinal: A `Tensor` of type `int32`.
      An int scalar tensor, representing the TPU device to use. This should be -1 when
      the Op is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    dtype: A `tf.DType`. The type of elements in the tensor.
    shape: A `tf.TensorShape` or list of `ints`. The shape of the tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'OutfeedDequeueV2', name, device_ordinal, 'dtype', dtype, 'shape', shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return outfeed_dequeue_v2_eager_fallback(device_ordinal, dtype=dtype, shape=shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    shape = _execute.make_shape(shape, 'shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('OutfeedDequeueV2', device_ordinal=device_ordinal, dtype=dtype, shape=shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'shape', _op.get_attr('shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('OutfeedDequeueV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result