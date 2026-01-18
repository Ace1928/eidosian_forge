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
def bucketize(input: _atypes.TensorFuzzingAnnotation[TV_Bucketize_T], boundaries, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Int32]:
    """Bucketizes 'input' based on 'boundaries'.

  For example, if the inputs are
      boundaries = [0, 10, 100]
      input = [[-5, 10000]
               [150,   10]
               [5,    100]]

  then the output will be
      output = [[0, 3]
                [3, 2]
                [1, 3]]

  Args:
    input: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
      Any shape of Tensor contains with int or float type.
    boundaries: A list of `floats`.
      A sorted list of floats gives the boundary of the buckets.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `int32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'Bucketize', name, input, 'boundaries', boundaries)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return bucketize_eager_fallback(input, boundaries=boundaries, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(boundaries, (list, tuple)):
        raise TypeError("Expected list for 'boundaries' argument to 'bucketize' Op, not %r." % boundaries)
    boundaries = [_execute.make_float(_f, 'boundaries') for _f in boundaries]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('Bucketize', input=input, boundaries=boundaries, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'boundaries', _op.get_attr('boundaries'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('Bucketize', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result