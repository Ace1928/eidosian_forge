from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.functions.elementary.integers import floor
def slice_of_slice(s, t):
    start1, stop1, step1 = s
    start2, stop2, step2 = t
    start = start1 + start2 * step1
    step = step1 * step2
    stop = start1 + step1 * stop2
    if stop > stop1:
        raise IndexError()
    return (start, stop, step)