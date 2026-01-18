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
def fake_param(dtype: TV_FakeParam_dtype, shape, name=None) -> _atypes.TensorFuzzingAnnotation[TV_FakeParam_dtype]:
    """  This op is used as a placeholder in If branch functions. It doesn't provide a
  valid output when run, so must either be removed (e.g. replaced with a
  function input) or guaranteed not to be used (e.g. if mirroring an
  intermediate output needed for the gradient computation of the other branch).

  Args:
    dtype: A `tf.DType`. The type of the output.
    shape: A `tf.TensorShape` or list of `ints`.
          The purported shape of the output. This is only used for shape inference;
          the output will not necessarily have this shape. Can be a partial shape.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'FakeParam', name, 'dtype', dtype, 'shape', shape)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return fake_param_eager_fallback(dtype=dtype, shape=shape, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    dtype = _execute.make_type(dtype, 'dtype')
    shape = _execute.make_shape(shape, 'shape')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('FakeParam', dtype=dtype, shape=shape, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'shape', _op.get_attr('shape'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('FakeParam', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result