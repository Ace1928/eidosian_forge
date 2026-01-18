from sage.all import (cached_method, real_part, imag_part, round, ceil, floor, log,
import itertools
def best_algdep_factor(z, degree):
    if hasattr(z, 'algebraic_dependency'):
        return z.algebraic_dependency(degree)
    else:
        P = z.algebraic_dependancy(degree)
        return sorted([p for p, e in P.factor()], key=lambda p: abs(p(z)))[0]