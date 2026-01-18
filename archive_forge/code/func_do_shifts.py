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
def do_shifts(fro, to, inc, dec):
    ops = []
    for i in range(len(fro)):
        if to[i] - fro[i] > 0:
            sh = inc
            ch = 1
        else:
            sh = dec
            ch = -1
        while to[i] != fro[i]:
            ops += [sh(fro, i)]
            fro[i] += ch
    return ops