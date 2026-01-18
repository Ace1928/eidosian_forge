from sympy.core.singleton import S
from sympy.printing.tableform import TableForm
from sympy.printing.latex import latex
from sympy.abc import x
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import raises
from textwrap import dedent
def neg_in_paren(x, i, j):
    if i % 2:
        return ('(%s)' if x < 0 else '%s') % x
    else:
        pass