import functools
import inspect
def func_accepts_var_args(func):
    """
    Return True if function 'func' accepts positional arguments *args.
    """
    return any((p for p in _get_callable_parameters(func) if p.kind == p.VAR_POSITIONAL))