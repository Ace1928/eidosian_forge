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
def detect_uppergamma(func):
    x = func.an[0]
    y, z = func.bm
    swapped = False
    if not _mod1((x - y).simplify()):
        swapped = True
        y, z = (z, y)
    if _mod1((x - z).simplify()) or x - z > 0:
        return None
    l = [y, x]
    if swapped:
        l = [x, y]
    return ({rho: y, a: x - y}, G_Function([x], [], l, []))