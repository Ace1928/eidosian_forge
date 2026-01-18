from sympy.concrete.summations import summation
from sympy.core.function import expand
from sympy.core.numbers import nan
from sympy.core.singleton import S
from sympy.core.symbol import Dummy as var
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.integers import floor
from sympy.matrices.dense import eye, Matrix, zeros
from sympy.printing.pretty.pretty import pretty_print as pprint
from sympy.simplify.simplify import simplify
from sympy.polys.domains import QQ
from sympy.polys.polytools import degree, LC, Poly, pquo, quo, prem, rem
from sympy.polys.polyerrors import PolynomialError
def rotate_l(L, k):
    """
    Rotates left by k. L is a row of a matrix or a list.

    """
    ll = list(L)
    if ll == []:
        return []
    for i in range(k):
        el = ll.pop(0)
        ll.insert(len(ll) - 1, el)
    return ll if isinstance(L, list) else Matrix([ll])