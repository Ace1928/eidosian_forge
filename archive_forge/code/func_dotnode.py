from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer, Rational, Float
from sympy.printing.repr import srepr
def dotnode(expr, styles=default_styles, labelfunc=str, pos=(), repeat=True):
    """ String defining a node

    Examples
    ========

    >>> from sympy.printing.dot import dotnode
    >>> from sympy.abc import x
    >>> print(dotnode(x))
    "Symbol('x')_()" ["color"="black", "label"="x", "shape"="ellipse"];
    """
    style = styleof(expr, styles)
    if isinstance(expr, Basic) and (not expr.is_Atom):
        label = str(expr.__class__.__name__)
    else:
        label = labelfunc(expr)
    style['label'] = label
    expr_str = purestr(expr)
    if repeat:
        expr_str += '_%s' % str(pos)
    return '"%s" [%s];' % (expr_str, attrprint(style))