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
@tf_export('complex_struct')
def complex_struct(n_a: int, n_b: int, t_c, name=None):
    """TODO: add doc.

  Args:
    n_a: An `int` that is `>= 0`.
    n_b: An `int` that is `>= 0`.
    t_c: A list of `tf.DTypes`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (a, b, c).

    a: A list of `n_a` `Tensor` objects with type `int32`.
    b: A list of `n_b` `Tensor` objects with type `int64`.
    c: A list of `Tensor` objects of type `t_c`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'ComplexStruct', name, 'n_a', n_a, 'n_b', n_b, 't_c', t_c)
            _result = _ComplexStructOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_complex_struct((n_a, n_b, t_c, name), None)
            if _result is not NotImplemented:
                return _result
            return complex_struct_eager_fallback(n_a=n_a, n_b=n_b, t_c=t_c, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(complex_struct, (), dict(n_a=n_a, n_b=n_b, t_c=t_c, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_complex_struct((n_a, n_b, t_c, name), None)
        if _result is not NotImplemented:
            return _result
    n_a = _execute.make_int(n_a, 'n_a')
    n_b = _execute.make_int(n_b, 'n_b')
    if not isinstance(t_c, (list, tuple)):
        raise TypeError("Expected list for 't_c' argument to 'complex_struct' Op, not %r." % t_c)
    t_c = [_execute.make_type(_t, 't_c') for _t in t_c]
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('ComplexStruct', n_a=n_a, n_b=n_b, t_c=t_c, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(complex_struct, (), dict(n_a=n_a, n_b=n_b, t_c=t_c, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('n_a', _op._get_attr_int('n_a'), 'n_b', _op._get_attr_int('n_b'), 't_c', _op.get_attr('t_c'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('ComplexStruct', _inputs_flat, _attrs, _result)
    _result = [_result[:n_a]] + _result[n_a:]
    _result = _result[:1] + [_result[1:1 + n_b]] + _result[1 + n_b:]
    _result = _result[:2] + [_result[2:]]
    _result = _ComplexStructOutput._make(_result)
    return _result