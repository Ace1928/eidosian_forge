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
def outfeed_dequeue_tuple_v2(device_ordinal: _atypes.TensorFuzzingAnnotation[_atypes.Int32], dtypes, shapes, name=None):
    """Retrieve multiple values from the computation outfeed. Device ordinal is a
tensor allowing dynamic outfeed.

  This operation will block indefinitely until data is available. Output `i`
  corresponds to XLA tuple element `i`.

  Args:
    device_ordinal: A `Tensor` of type `int32`.
      An int scalar tensor, representing the TPU device to use. This should be -1 when
      the Op is running on a TPU device, and >= 0 when the Op is running on the CPU
      device.
    dtypes: A list of `tf.DTypes` that has length `>= 1`.
      The element types of each element in `outputs`.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `outputs`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `dtypes`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'OutfeedDequeueTupleV2', name, device_ordinal, 'dtypes', dtypes, 'shapes', shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return outfeed_dequeue_tuple_v2_eager_fallback(device_ordinal, dtypes=dtypes, shapes=shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(dtypes, (list, tuple)):
        raise TypeError("Expected list for 'dtypes' argument to 'outfeed_dequeue_tuple_v2' Op, not %r." % dtypes)
    dtypes = [_execute.make_type(_t, 'dtypes') for _t in dtypes]
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'outfeed_dequeue_tuple_v2' Op, not %r." % shapes)
    shapes = [_execute.make_shape(_s, 'shapes') for _s in shapes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('OutfeedDequeueTupleV2', device_ordinal=device_ordinal, dtypes=dtypes, shapes=shapes, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('dtypes', _op.get_attr('dtypes'), 'shapes', _op.get_attr('shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('OutfeedDequeueTupleV2', _inputs_flat, _attrs, _result)
    return _result