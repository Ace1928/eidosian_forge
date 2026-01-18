from __future__ import annotations
from sympy.core import Symbol
from sympy.utilities.iterables import iterable
def build_product_order(arg, gens):
    """
    Build a monomial order on ``gens``.

    ``arg`` should be a tuple of iterables. The first element of each iterable
    should be a string or monomial order (will be passed to monomial_key),
    the others should be subsets of the generators. This function will build
    the corresponding product order.

    For example, build a product of two grlex orders:

    >>> from sympy.polys.orderings import build_product_order
    >>> from sympy.abc import x, y, z, t

    >>> O = build_product_order((("grlex", x, y), ("grlex", z, t)), [x, y, z, t])
    >>> O((1, 2, 3, 4))
    ((3, (1, 2)), (7, (3, 4)))

    """
    gens2idx = {}
    for i, g in enumerate(gens):
        gens2idx[g] = i
    order = []
    for expr in arg:
        name = expr[0]
        var = expr[1:]

        def makelambda(var):
            return _ItemGetter((gens2idx[g] for g in var))
        order.append((monomial_key(name), makelambda(var)))
    return ProductOrder(*order)