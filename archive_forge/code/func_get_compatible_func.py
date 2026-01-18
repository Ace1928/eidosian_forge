import collections
import itertools
import typing  # pylint: disable=unused-import (used in doctests)
from tensorflow.python.framework import _pywrap_python_api_dispatcher as _api_dispatcher
from tensorflow.python.framework import ops
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_export as tf_export_lib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import traceback_utils
from tensorflow.python.util import type_annotations
from tensorflow.python.util.tf_export import tf_export
def get_compatible_func(op, func):
    """Returns a compatible function.

  Args:
    op: a callable with whose signature the returned function is compatible.
    func: a callable which is called by the returned function.

  Returns:
    a compatible function, which conducts the actions of `func` but can
    be called like `op`, given that:
      - the list of required arguments in `func` and `op` are the same.
      - there is no override of the default arguments of `op` that are not
        supported by `func`.
  """
    op_signature = _remove_annotation(tf_inspect.signature(op))
    func_signature = _remove_annotation(tf_inspect.signature(func))
    if op_signature == func_signature:
        return func
    op_pos_names = _get_required_param_names(op_signature)
    func_pos_names = _get_required_param_names(func_signature)
    if op_pos_names != func_pos_names:
        raise AssertionError(f"The decorated function's non-default arguments must be identical to that of the overridden op. func has {func_pos_names}. op has {op_pos_names}.")
    func_missing_params = {}
    for name in set(op_signature.parameters.keys()) - set(func_signature.parameters.keys()):
        p = op_signature.parameters[name]
        if p.default is p.empty:
            raise AssertionError(f"The decorated function's signature must implement all of the non-default arguments of the overridden op. Argument `{name}` is unimplemented.")
        func_missing_params[name] = p

    def compatible_func(*args, **kwargs):
        bound = op_signature.bind(*args, **kwargs)
        for name, param in func_missing_params.items():
            if name not in bound.arguments:
                continue
            value = bound.arguments.pop(name)
            if value is not param.default:
                raise AssertionError(f'Dispatched op is called with argument `{name}` set to a non-default value, which is not supported by the decorated function')
        return func(*bound.args, **bound.kwargs)
    return compatible_func