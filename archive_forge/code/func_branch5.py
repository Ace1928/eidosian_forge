from sympy.strategies.branch.core import (
def branch5(x):
    if 0 < x < 5:
        yield (x - 1)
    elif 5 < x < 10:
        yield (x + 1)
    elif x == 5:
        yield (x + 1)
        yield (x - 1)
    else:
        yield x