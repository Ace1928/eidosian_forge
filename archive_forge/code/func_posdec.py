from sympy.strategies.branch.core import (
def posdec(x):
    if x > 0:
        yield (x - 1)
    else:
        yield x