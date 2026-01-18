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
@tf_export('func_list_attr')
def func_list_attr(f, name=None):
    """TODO: add doc.

  Args:
    f: A list of functions decorated with @Defun.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'FuncListAttr', name, 'f', f)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            _result = _dispatcher_for_func_list_attr((f, name), None)
            if _result is not NotImplemented:
                return _result
            return func_list_attr_eager_fallback(f=f, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
        except (TypeError, ValueError):
            _result = _dispatch.dispatch(func_list_attr, (), dict(f=f, name=name))
            if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
                return _result
            raise
    else:
        _result = _dispatcher_for_func_list_attr((f, name), None)
        if _result is not NotImplemented:
            return _result
    if not isinstance(f, (list, tuple)):
        raise TypeError("Expected list for 'f' argument to 'func_list_attr' Op, not %r." % f)
    try:
        _, _, _op, _outputs = _op_def_library._apply_op_helper('FuncListAttr', f=f, name=name)
    except (TypeError, ValueError):
        _result = _dispatch.dispatch(func_list_attr, (), dict(f=f, name=name))
        if _result is not _dispatch.OpDispatcher.NOT_SUPPORTED:
            return _result
        raise
    return _op