from .basic import Basic
from .sorting import ordered
from .sympify import sympify
from sympy.utilities.iterables import iterable
def iterfreeargs(expr, _first=True):
    """Yield the args of a Basic object in a breadth-first traversal.
    Depth-traversal stops if `arg.args` is either empty or is not
    an iterable. The bound objects of an expression will be returned
    as canonical variables.

    Examples
    ========

    >>> from sympy import Integral, Function
    >>> from sympy.abc import x
    >>> f = Function('f')
    >>> from sympy.core.traversal import iterfreeargs
    >>> list(iterfreeargs(Integral(f(x), (f(x), 1))))
    [Integral(f(x), (f(x), 1)), 1]

    See Also
    ========
    iterargs, preorder_traversal
    """
    args = [expr]
    for i in args:
        yield i
        if _first and hasattr(i, 'bound_symbols'):
            void = i.canonical_variables.values()
            for i in iterfreeargs(i.as_dummy(), _first=False):
                if not i.has(*void):
                    yield i
        try:
            args.extend(i.args)
        except TypeError:
            pass