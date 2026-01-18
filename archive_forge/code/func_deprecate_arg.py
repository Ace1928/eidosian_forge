from __future__ import annotations
import functools
import inspect
import warnings
from collections.abc import Callable
from typing import Any, Type
def deprecate_arg(name: str, *, since: str, additional_msg: str | None=None, deprecation_description: str | None=None, pending: bool=False, package_name: str='qiskit', new_alias: str | None=None, predicate: Callable[[Any], bool] | None=None, removal_timeline: str='no earlier than 3 months after the release date'):
    """Decorator to indicate an argument has been deprecated in some way.

    This decorator may be used multiple times on the same function, once per deprecated argument.
    It should be placed beneath other decorators like ``@staticmethod`` and property decorators.

    Args:
        name: The name of the deprecated argument.
        since: The version the deprecation started at. If the deprecation is pending, set
            the version to when that started; but later, when switching from pending to
            deprecated, update `since` to the new version.
        deprecation_description: What is being deprecated? E.g. "Setting my_func()'s `my_arg`
            argument to `None`." If not set, will default to "{func_name}'s argument `{name}`".
        additional_msg: Put here any additional information, such as what to use instead
            (if new_alias is not set). For example, "Instead, use the argument `new_arg`,
            which is similar but does not impact the circuit's setup."
        pending: Set to `True` if the deprecation is still pending.
        package_name: The PyPI package name, e.g. "qiskit-nature".
        new_alias: If the arg has simply been renamed, set this to the new name. The decorator will
            dynamically update the `kwargs` so that when the user sets the old arg, it will be
            passed in as the `new_alias` arg.
        predicate: Only log the runtime warning if the predicate returns True. This is useful to
            deprecate certain values or types for an argument, e.g.
            `lambda my_arg: isinstance(my_arg, dict)`. Regardless of if a predicate is set, the
            runtime warning will only log when the user specifies the argument.
        removal_timeline: How soon can this deprecation be removed? Expects a value
            like "no sooner than 6 months after the latest release" or "in release 9.99".

    Returns:
        Callable: The decorated callable.
    """

    def decorator(func):
        func_name = f'{func.__module__}.{func.__qualname__}()'
        deprecated_entity = deprecation_description or f"``{func_name}``'s argument ``{name}``"
        if new_alias:
            alias_msg = f'Instead, use the argument ``{new_alias}``, which behaves identically.'
            if additional_msg:
                final_additional_msg = f'{alias_msg}. {additional_msg}'
            else:
                final_additional_msg = alias_msg
        else:
            final_additional_msg = additional_msg
        msg, category = _write_deprecation_msg(deprecated_entity=deprecated_entity, package_name=package_name, since=since, pending=pending, additional_msg=final_additional_msg, removal_timeline=removal_timeline)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _maybe_warn_and_rename_kwarg(args, kwargs, func_name=func_name, original_func_co_varnames=wrapper.__original_func_co_varnames, old_arg_name=name, new_alias=new_alias, warning_msg=msg, category=category, predicate=predicate)
            return func(*args, **kwargs)
        if hasattr(func, '__original_func_co_varnames'):
            wrapper.__original_func_co_varnames = func.__original_func_co_varnames
        else:
            wrapper.__original_func_co_varnames = func.__code__.co_varnames
            param_kinds = {param.kind for param in inspect.signature(func).parameters.values()}
            if inspect.Parameter.VAR_POSITIONAL in param_kinds:
                raise ValueError('@deprecate_arg cannot be used with functions that take variable *args. Use warnings.warn() directly instead.')
        add_deprecation_to_docstring(wrapper, msg, since=since, pending=pending)
        return wrapper
    return decorator