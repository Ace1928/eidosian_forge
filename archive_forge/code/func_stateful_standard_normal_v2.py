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
def stateful_standard_normal_v2(resource: _atypes.TensorFuzzingAnnotation[_atypes.Resource], algorithm: _atypes.TensorFuzzingAnnotation[_atypes.Int64], shape: _atypes.TensorFuzzingAnnotation[TV_StatefulStandardNormalV2_shape_dtype], dtype: TV_StatefulStandardNormalV2_dtype=_dtypes.float32, name=None) -> _atypes.TensorFuzzingAnnotation[TV_StatefulStandardNormalV2_dtype]:
    """Outputs random values from a normal distribution.

  The generated values will have mean 0 and standard deviation 1.

  Args:
    resource: A `Tensor` of type `resource`.
      The handle of the resource variable that stores the state of the RNG.
    algorithm: A `Tensor` of type `int64`. The RNG algorithm.
    shape: A `Tensor`. The shape of the output tensor.
    dtype: An optional `tf.DType`. Defaults to `tf.float32`.
      The type of the output.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatefulStandardNormalV2', name, resource, algorithm, shape, 'dtype', dtype)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateful_standard_normal_v2_eager_fallback(resource, algorithm, shape, dtype=dtype, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if dtype is None:
        dtype = _dtypes.float32
    dtype = _execute.make_type(dtype, 'dtype')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatefulStandardNormalV2', resource=resource, algorithm=algorithm, shape=shape, dtype=dtype, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtype', _op._get_attr_type('dtype'), 'shape_dtype', _op._get_attr_type('shape_dtype'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatefulStandardNormalV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result