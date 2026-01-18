from .core import unify, reify  # type: ignore[attr-defined]
from .dispatch import dispatch
def reify_object(o, s):
    """ Reify a Python object with a substitution
    >>> # xdoctest: +SKIP
    >>> class Foo(object):
    ...     def __init__(self, a, b):
    ...         self.a = a
    ...         self.b = b
    ...     def __str__(self):
    ...         return "Foo(%s, %s)"%(str(self.a), str(self.b))
    >>> x = var('x')
    >>> f = Foo(1, x)
    >>> print(f)
    Foo(1, ~x)
    >>> print(reify_object(f, {x: 2}))
    Foo(1, 2)
    """
    if hasattr(o, '__slots__'):
        return _reify_object_slots(o, s)
    else:
        return _reify_object_dict(o, s)