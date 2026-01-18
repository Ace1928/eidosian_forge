import string
from ..sage_helper import _within_sage, sage_method
def minimum_exponents(elts):
    exps = iter(join_lists((uniform_poly_exponents(p) for p in elts)))
    mins = list(next(exps))
    n = len(mins)
    for e in exps:
        for i in range(n):
            if e[i] < mins[i]:
                mins[i] = e[i]
    return mins