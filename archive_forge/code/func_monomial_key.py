from __future__ import annotations
from sympy.core import Symbol
from sympy.utilities.iterables import iterable
def monomial_key(order=None, gens=None):
    """
    Return a function defining admissible order on monomials.

    The result of a call to :func:`monomial_key` is a function which should
    be used as a key to :func:`sorted` built-in function, to provide order
    in a set of monomials of the same length.

    Currently supported monomial orderings are:

    1. lex       - lexicographic order (default)
    2. grlex     - graded lexicographic order
    3. grevlex   - reversed graded lexicographic order
    4. ilex, igrlex, igrevlex - the corresponding inverse orders

    If the ``order`` input argument is not a string but has ``__call__``
    attribute, then it will pass through with an assumption that the
    callable object defines an admissible order on monomials.

    If the ``gens`` input argument contains a list of generators, the
    resulting key function can be used to sort SymPy ``Expr`` objects.

    """
    if order is None:
        order = lex
    if isinstance(order, Symbol):
        order = str(order)
    if isinstance(order, str):
        try:
            order = _monomial_key[order]
        except KeyError:
            raise ValueError("supported monomial orderings are 'lex', 'grlex' and 'grevlex', got %r" % order)
    if hasattr(order, '__call__'):
        if gens is not None:

            def _order(expr):
                return order(expr.as_poly(*gens).degree_list())
            return _order
        return order
    else:
        raise ValueError('monomial ordering specification must be a string or a callable, got %s' % order)