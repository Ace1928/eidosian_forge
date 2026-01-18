from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer, Rational, Float
from sympy.printing.repr import srepr
def styleof(expr, styles=default_styles):
    """ Merge style dictionaries in order

    Examples
    ========

    >>> from sympy import Symbol, Basic, Expr, S
    >>> from sympy.printing.dot import styleof
    >>> styles = [(Basic, {'color': 'blue', 'shape': 'ellipse'}),
    ...           (Expr,  {'color': 'black'})]

    >>> styleof(Basic(S(1)), styles)
    {'color': 'blue', 'shape': 'ellipse'}

    >>> x = Symbol('x')
    >>> styleof(x + 1, styles)  # this is an Expr
    {'color': 'black', 'shape': 'ellipse'}
    """
    style = {}
    for typ, sty in styles:
        if isinstance(expr, typ):
            style.update(sty)
    return style