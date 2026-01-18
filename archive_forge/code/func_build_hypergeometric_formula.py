from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def build_hypergeometric_formula(func):
    """
    Create a formula object representing the hypergeometric function ``func``.

    """
    z = Dummy('z')
    if func.ap:
        afactors = [_x + a for a in func.ap]
        bfactors = [_x + b - 1 for b in func.bq]
        expr = _x * Mul(*bfactors) - z * Mul(*afactors)
        poly = Poly(expr, _x)
        n = poly.degree()
        basis = []
        M = zeros(n)
        for k in range(n):
            a = func.ap[0] + k
            basis += [hyper([a] + list(func.ap[1:]), func.bq, z)]
            if k < n - 1:
                M[k, k] = -a
                M[k, k + 1] = a
        B = Matrix(basis)
        C = Matrix([[1] + [0] * (n - 1)])
        derivs = [eye(n)]
        for k in range(n):
            derivs.append(M * derivs[k])
        l = poly.all_coeffs()
        l.reverse()
        res = [0] * n
        for k, c in enumerate(l):
            for r, d in enumerate(C * derivs[k]):
                res[r] += c * d
        for k, c in enumerate(res):
            M[n - 1, k] = -c / derivs[n - 1][0, n - 1] / poly.all_coeffs()[0]
        return Formula(func, z, None, [], B, C, M)
    else:
        basis = []
        bq = list(func.bq[:])
        for i in range(len(bq)):
            basis += [hyper([], bq, z)]
            bq[i] += 1
        basis += [hyper([], bq, z)]
        B = Matrix(basis)
        n = len(B)
        C = Matrix([[1] + [0] * (n - 1)])
        M = zeros(n)
        M[0, n - 1] = z / Mul(*func.bq)
        for k in range(1, n):
            M[k, k - 1] = func.bq[k - 1]
            M[k, k] = -func.bq[k - 1]
        return Formula(func, z, None, [], B, C, M)