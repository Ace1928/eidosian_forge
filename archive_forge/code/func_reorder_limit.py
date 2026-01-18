from sympy.concrete.expr_with_limits import ExprWithLimits
from sympy.core.singleton import S
from sympy.core.relational import Eq
def reorder_limit(expr, x, y):
    """
        Interchange two limit tuples of a Sum or Product expression.

        Explanation
        ===========

        ``expr.reorder_limit(x, y)`` interchanges two limit tuples. The
        arguments ``x`` and ``y`` are integers corresponding to the index
        variables of the two limits which are to be interchanged. The
        expression ``expr`` has to be either a Sum or a Product.

        Examples
        ========

        >>> from sympy.abc import x, y, z, a, b, c, d, e, f
        >>> from sympy import Sum, Product

        >>> Sum(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder_limit(0, 2)
        Sum(x*y*z, (z, e, f), (y, c, d), (x, a, b))
        >>> Sum(x**2, (x, a, b), (x, c, d)).reorder_limit(1, 0)
        Sum(x**2, (x, c, d), (x, a, b))

        >>> Product(x*y*z, (x, a, b), (y, c, d), (z, e, f)).reorder_limit(0, 2)
        Product(x*y*z, (z, e, f), (y, c, d), (x, a, b))

        See Also
        ========

        index, reorder, sympy.concrete.summations.Sum.reverse_order,
        sympy.concrete.products.Product.reverse_order
        """
    var = {limit[0] for limit in expr.limits}
    limit_x = expr.limits[x]
    limit_y = expr.limits[y]
    if len(set(limit_x[1].free_symbols).intersection(var)) == 0 and len(set(limit_x[2].free_symbols).intersection(var)) == 0 and (len(set(limit_y[1].free_symbols).intersection(var)) == 0) and (len(set(limit_y[2].free_symbols).intersection(var)) == 0):
        limits = []
        for i, limit in enumerate(expr.limits):
            if i == x:
                limits.append(limit_y)
            elif i == y:
                limits.append(limit_x)
            else:
                limits.append(limit)
        return type(expr)(expr.function, *limits)
    else:
        raise ReorderError(expr, 'could not interchange the two limits specified')