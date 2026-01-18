import functools
import inspect
def func_accepts_kwargs(func):
    """Return True if function 'func' accepts keyword arguments **kwargs."""
    return any((p for p in _get_callable_parameters(func) if p.kind == p.VAR_KEYWORD))