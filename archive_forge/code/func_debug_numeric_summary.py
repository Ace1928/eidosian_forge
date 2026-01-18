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
def debug_numeric_summary(input: _atypes.TensorFuzzingAnnotation[TV_DebugNumericSummary_T], device_name: str='', tensor_name: str='', debug_urls=[], lower_bound: float=float('-inf'), upper_bound: float=float('inf'), mute_if_healthy: bool=False, gated_grpc: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Float64]:
    """Debug Numeric Summary Op.

  Provide a basic summary of numeric value types, range and distribution.

  output: A double tensor of shape [14 + nDimensions], where nDimensions is the
    number of dimensions of the tensor's shape. The elements of output are:
    [0]: is initialized (1.0) or not (0.0).
    [1]: total number of elements
    [2]: NaN element count
    [3]: generalized -inf count: elements <= lower_bound. lower_bound is -inf by
      default.
    [4]: negative element count (excluding -inf), if lower_bound is the default
      -inf. Otherwise, this is the count of elements > lower_bound and < 0.
    [5]: zero element count
    [6]: positive element count (excluding +inf), if upper_bound is the default
      +inf. Otherwise, this is the count of elements < upper_bound and > 0.
    [7]: generalized +inf count, elements >= upper_bound. upper_bound is +inf by
      default.
  Output elements [1:8] are all zero, if the tensor is uninitialized.
    [8]: minimum of all non-inf and non-NaN elements.
         If uninitialized or no such element exists: +inf.
    [9]: maximum of all non-inf and non-NaN elements.
         If uninitialized or no such element exists: -inf.
    [10]: mean of all non-inf and non-NaN elements.
          If uninitialized or no such element exists: NaN.
    [11]: variance of all non-inf and non-NaN elements.
          If uninitialized or no such element exists: NaN.
    [12]: Data type of the tensor encoded as an enum integer. See the DataType
          proto for more details.
    [13]: Number of dimensions of the tensor (ndims).
    [14+]: Sizes of the dimensions.

  Args:
    input: A `Tensor`. Input tensor, non-Reference type.
    device_name: An optional `string`. Defaults to `""`.
    tensor_name: An optional `string`. Defaults to `""`.
      Name of the input tensor.
    debug_urls: An optional list of `strings`. Defaults to `[]`.
      List of URLs to debug targets, e.g.,
        file:///foo/tfdbg_dump, grpc:://localhost:11011.
    lower_bound: An optional `float`. Defaults to `float('-inf')`.
      (float) The lower bound <= which values will be included in the
        generalized -inf count. Default: -inf.
    upper_bound: An optional `float`. Defaults to `float('inf')`.
      (float) The upper bound >= which values will be included in the
        generalized +inf count. Default: +inf.
    mute_if_healthy: An optional `bool`. Defaults to `False`.
      (bool) Do not send data to the debug URLs unless at least one
        of elements [2], [3] and [7] (i.e., the nan count and the generalized -inf and
        inf counts) is non-zero.
    gated_grpc: An optional `bool`. Defaults to `False`.
      Whether this op will be gated. If any of the debug_urls of this
        debug node is of the grpc:// scheme, when the value of this attribute is set
        to True, the data will not actually be sent via the grpc stream unless this
        debug op has been enabled at the debug_url. If all of the debug_urls of this
        debug node are of the grpc:// scheme and the debug op is enabled at none of
        them, the output will be an empty Tensor.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float64`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'DebugNumericSummary', name, input, 'device_name', device_name, 'tensor_name', tensor_name, 'debug_urls', debug_urls, 'lower_bound', lower_bound, 'upper_bound', upper_bound, 'mute_if_healthy', mute_if_healthy, 'gated_grpc', gated_grpc)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return debug_numeric_summary_eager_fallback(input, device_name=device_name, tensor_name=tensor_name, debug_urls=debug_urls, lower_bound=lower_bound, upper_bound=upper_bound, mute_if_healthy=mute_if_healthy, gated_grpc=gated_grpc, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if device_name is None:
        device_name = ''
    device_name = _execute.make_str(device_name, 'device_name')
    if tensor_name is None:
        tensor_name = ''
    tensor_name = _execute.make_str(tensor_name, 'tensor_name')
    if debug_urls is None:
        debug_urls = []
    if not isinstance(debug_urls, (list, tuple)):
        raise TypeError("Expected list for 'debug_urls' argument to 'debug_numeric_summary' Op, not %r." % debug_urls)
    debug_urls = [_execute.make_str(_s, 'debug_urls') for _s in debug_urls]
    if lower_bound is None:
        lower_bound = float('-inf')
    lower_bound = _execute.make_float(lower_bound, 'lower_bound')
    if upper_bound is None:
        upper_bound = float('inf')
    upper_bound = _execute.make_float(upper_bound, 'upper_bound')
    if mute_if_healthy is None:
        mute_if_healthy = False
    mute_if_healthy = _execute.make_bool(mute_if_healthy, 'mute_if_healthy')
    if gated_grpc is None:
        gated_grpc = False
    gated_grpc = _execute.make_bool(gated_grpc, 'gated_grpc')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('DebugNumericSummary', input=input, device_name=device_name, tensor_name=tensor_name, debug_urls=debug_urls, lower_bound=lower_bound, upper_bound=upper_bound, mute_if_healthy=mute_if_healthy, gated_grpc=gated_grpc, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('T', _op._get_attr_type('T'), 'device_name', _op.get_attr('device_name'), 'tensor_name', _op.get_attr('tensor_name'), 'debug_urls', _op.get_attr('debug_urls'), 'lower_bound', _op.get_attr('lower_bound'), 'upper_bound', _op.get_attr('upper_bound'), 'mute_if_healthy', _op._get_attr_bool('mute_if_healthy'), 'gated_grpc', _op._get_attr_bool('gated_grpc'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('DebugNumericSummary', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result