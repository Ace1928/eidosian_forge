def finite_diff(expression, variable, increment=1):
    """
    Takes as input a polynomial expression and the variable used to construct
    it and returns the difference between function's value when the input is
    incremented to 1 and the original function value. If you want an increment
    other than one supply it as a third argument.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.series.kauers import finite_diff
    >>> finite_diff(x**2, x)
    2*x + 1
    >>> finite_diff(y**3 + 2*y**2 + 3*y + 4, y)
    3*y**2 + 7*y + 6
    >>> finite_diff(x**2 + 3*x + 8, x, 2)
    4*x + 10
    >>> finite_diff(z**3 + 8*z, z, 3)
    9*z**2 + 27*z + 51
    """
    expression = expression.expand()
    expression2 = expression.subs(variable, variable + increment)
    expression2 = expression2.expand()
    return expression2 - expression