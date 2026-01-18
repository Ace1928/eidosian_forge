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
@_dispatch.add_fallback_dispatch_list
@_dispatch.add_type_based_api_dispatcher
@tf_export('nn.local_response_normalization', 'nn.lrn')
def lrn(input: _atypes.TensorFuzzingAnnotation[TV_LRN_T], depth_radius: int=5, bias: float=1, alpha: float=1, beta: float=0.5, name=None) -> _atypes.TensorFuzzingAnnotation[TV_LRN_T]:
    """Local Response Normalization.

  The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
  dimension), and each vector is normalized independently.  Within a given vector,
  each component is divided by the weighted, squared sum of inputs within
  `depth_radius`.  In detail,

      sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
      output = input / (bias + alpha * sqr_sum) ** beta

  For details, see [Krizhevsky et al., ImageNet classification with deep
  convolutional neural networks (NIPS 2012)](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

  Args:
    input: A `Tensor`. Must be one of the following types: `half`, `bfloat16`, `float32`.
      4-D.
    depth_radius: An optional `int`. Defaults to `5`.
      0-D.  Half-width of the 1-D normalization window.
    bias: An optional `float`. Defaults to `1`.
      An offset (usually positive to avoid dividing by 0).
    alpha: An optional `float`. Defaults to `1`.
      A scale factor, usually positive.
    beta: An optional `float`. Defaults to `0.5`. An exponent.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'LRN', name, input, 'depth_radius', depth_radius, 'bias', bias, 'alpha', alpha, 'beta', beta)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_lrn((input, depth_radius, bias, alpha, beta, name), None)
            if _result is not NotImplemented:
                return _result
            return lrn_eager_fallback(input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(lrn, (), dict(input=input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_lrn((input, depth_radius, bias, alpha, beta, name), None)
        if _result is not NotImplemented:
            return _result
    if depth_radius is None:
        depth_radius = 5
    depth_radius = _execute.make_int(depth_radius, 'depth_radius')
    if bias is None:
        bias = 1
    bias = _execute.make_float(bias, 'bias')
    if alpha is None:
        alpha = 1
    alpha = _execute.make_float(alpha, 'alpha')
    if beta is None:
        beta = 0.5
    beta = _execute.make_float(beta, 'beta')
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('LRN', input=input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(lrn, (), dict(input=input, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('depth_radius', _op._get_attr_int('depth_radius'), 'bias', _op.get_attr('bias'), 'alpha', _op.get_attr('alpha'), 'beta', _op.get_attr('beta'), 'T', _op._get_attr_type('T'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('LRN', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result