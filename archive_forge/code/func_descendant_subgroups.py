from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.combinatorics.free_groups import (FreeGroup, FreeGroupElement,
from sympy.combinatorics.rewritingsystem import RewritingSystem
from sympy.combinatorics.coset_table import (CosetTable,
from sympy.combinatorics import PermutationGroup
from sympy.matrices.normalforms import invariant_factors
from sympy.matrices import Matrix
from sympy.polys.polytools import gcd
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.magic import pollute
from itertools import product
def descendant_subgroups(S, C, R1_c_list, x, R2, N, Y):
    A_dict = C.A_dict
    A_dict_inv = C.A_dict_inv
    if C.is_complete():
        for w, alpha in product(R2, C.omega):
            if not C.scan_check(alpha, w):
                return
        S.append(C)
    else:
        for alpha, x in product(range(len(C.table)), C.A):
            if C.table[alpha][A_dict[x]] is None:
                undefined_coset, undefined_gen = (alpha, x)
                break
        reach = C.omega + [C.n]
        for beta in reach:
            if beta < N:
                if beta == C.n or C.table[beta][A_dict_inv[undefined_gen]] is None:
                    try_descendant(S, C, R1_c_list, R2, N, undefined_coset, undefined_gen, beta, Y)