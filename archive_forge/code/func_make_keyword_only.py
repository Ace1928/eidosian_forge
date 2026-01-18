import contextlib
import functools
import inspect
import math
import warnings
def make_keyword_only(since, name, func=None):
    """
    Decorator indicating that passing parameter *name* (or any of the following
    ones) positionally to *func* is being deprecated.

    When used on a method that has a pyplot wrapper, this should be the
    outermost decorator, so that :file:`boilerplate.py` can access the original
    signature.
    """
    decorator = functools.partial(make_keyword_only, since, name)
    if func is None:
        return decorator
    signature = inspect.signature(func)
    POK = inspect.Parameter.POSITIONAL_OR_KEYWORD
    KWO = inspect.Parameter.KEYWORD_ONLY
    assert name in signature.parameters and signature.parameters[name].kind == POK, f'Matplotlib internal error: {name!r} must be a positional-or-keyword parameter for {func.__name__}()'
    names = [*signature.parameters]
    name_idx = names.index(name)
    kwonly = [name for name in names[name_idx:] if signature.parameters[name].kind == POK]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > name_idx:
            warn_deprecated(since, message='Passing the %(name)s %(obj_type)s positionally is deprecated since Matplotlib %(since)s; the parameter will become keyword-only %(removal)s.', name=name, obj_type=f'parameter of {func.__name__}()')
        return func(*args, **kwargs)
    wrapper.__signature__ = signature.replace(parameters=[param.replace(kind=KWO) if param.name in kwonly else param for param in signature.parameters.values()])
    DECORATORS[wrapper] = decorator
    return wrapper