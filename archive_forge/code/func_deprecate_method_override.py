import contextlib
import functools
import inspect
import math
import warnings
def deprecate_method_override(method, obj, *, allow_empty=False, **kwargs):
    """
    Return ``obj.method`` with a deprecation if it was overridden, else None.

    Parameters
    ----------
    method
        An unbound method, i.e. an expression of the form
        ``Class.method_name``.  Remember that within the body of a method, one
        can always use ``__class__`` to refer to the class that is currently
        being defined.
    obj
        Either an object of the class where *method* is defined, or a subclass
        of that class.
    allow_empty : bool, default: False
        Whether to allow overrides by "empty" methods without emitting a
        warning.
    **kwargs
        Additional parameters passed to `warn_deprecated` to generate the
        deprecation warning; must at least include the "since" key.
    """

    def empty():
        pass

    def empty_with_docstring():
        """doc"""
    name = method.__name__
    bound_child = getattr(obj, name)
    bound_base = method if isinstance(bound_child, type(empty)) and isinstance(obj, type) else method.__get__(obj)
    if bound_child != bound_base and (not allow_empty or getattr(getattr(bound_child, '__code__', None), 'co_code', None) not in [empty.__code__.co_code, empty_with_docstring.__code__.co_code]):
        warn_deprecated(**{'name': name, 'obj_type': 'method', **kwargs})
        return bound_child
    return None