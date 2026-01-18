from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def best_origin(a, b, lineseg, expr):
    """Helper method for polytope_integrate. Currently not used in the main
    algorithm.

    Explanation
    ===========

    Returns a point on the lineseg whose vector inner product with the
    divergence of `expr` yields an expression with the least maximum
    total power.

    Parameters
    ==========

    a :
        Hyperplane parameter denoting direction.
    b :
        Hyperplane parameter denoting distance.
    lineseg :
        Line segment on which to find the origin.
    expr :
        The expression which determines the best point.

    Algorithm(currently works only for 2D use case)
    ===============================================

    1 > Firstly, check for edge cases. Here that would refer to vertical
        or horizontal lines.

    2 > If input expression is a polynomial containing more than one generator
        then find out the total power of each of the generators.

        x**2 + 3 + x*y + x**4*y**5 ---> {x: 7, y: 6}

        If expression is a constant value then pick the first boundary point
        of the line segment.

    3 > First check if a point exists on the line segment where the value of
        the highest power generator becomes 0. If not check if the value of
        the next highest becomes 0. If none becomes 0 within line segment
        constraints then pick the first boundary point of the line segment.
        Actually, any point lying on the segment can be picked as best origin
        in the last case.

    Examples
    ========

    >>> from sympy.integrals.intpoly import best_origin
    >>> from sympy.abc import x, y
    >>> from sympy import Point, Segment2D
    >>> l = Segment2D(Point(0, 3), Point(1, 1))
    >>> expr = x**3*y**7
    >>> best_origin((2, 1), 3, l, expr)
    (0, 3.0)
    """
    a1, b1 = lineseg.points[0]

    def x_axis_cut(ls):
        """Returns the point where the input line segment
        intersects the x-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        p, q = ls.points
        if p.y.is_zero:
            return tuple(p)
        elif q.y.is_zero:
            return tuple(q)
        elif p.y / q.y < S.Zero:
            return (p.y * (p.x - q.x) / (q.y - p.y) + p.x, S.Zero)
        else:
            return ()

    def y_axis_cut(ls):
        """Returns the point where the input line segment
        intersects the y-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        p, q = ls.points
        if p.x.is_zero:
            return tuple(p)
        elif q.x.is_zero:
            return tuple(q)
        elif p.x / q.x < S.Zero:
            return (S.Zero, p.x * (p.y - q.y) / (q.x - p.x) + p.y)
        else:
            return ()
    gens = (x, y)
    power_gens = {}
    for i in gens:
        power_gens[i] = S.Zero
    if len(gens) > 1:
        if len(gens) == 2:
            if a[0] == 0:
                if y_axis_cut(lineseg):
                    return (S.Zero, b / a[1])
                else:
                    return (a1, b1)
            elif a[1] == 0:
                if x_axis_cut(lineseg):
                    return (b / a[0], S.Zero)
                else:
                    return (a1, b1)
        if isinstance(expr, Expr):
            if expr.is_Add:
                for monomial in expr.args:
                    if monomial.is_Pow:
                        if monomial.args[0] in gens:
                            power_gens[monomial.args[0]] += monomial.args[1]
                    else:
                        for univariate in monomial.args:
                            term_type = len(univariate.args)
                            if term_type == 0 and univariate in gens:
                                power_gens[univariate] += 1
                            elif term_type == 2 and univariate.args[0] in gens:
                                power_gens[univariate.args[0]] += univariate.args[1]
            elif expr.is_Mul:
                for term in expr.args:
                    term_type = len(term.args)
                    if term_type == 0 and term in gens:
                        power_gens[term] += 1
                    elif term_type == 2 and term.args[0] in gens:
                        power_gens[term.args[0]] += term.args[1]
            elif expr.is_Pow:
                power_gens[expr.args[0]] = expr.args[1]
            elif expr.is_Symbol:
                power_gens[expr] += 1
        else:
            return (a1, b1)
        power_gens = sorted(power_gens.items(), key=lambda k: str(k[0]))
        if power_gens[0][1] >= power_gens[1][1]:
            if y_axis_cut(lineseg):
                x0 = (S.Zero, b / a[1])
            elif x_axis_cut(lineseg):
                x0 = (b / a[0], S.Zero)
            else:
                x0 = (a1, b1)
        elif x_axis_cut(lineseg):
            x0 = (b / a[0], S.Zero)
        elif y_axis_cut(lineseg):
            x0 = (S.Zero, b / a[1])
        else:
            x0 = (a1, b1)
    else:
        x0 = b / a[0]
    return x0