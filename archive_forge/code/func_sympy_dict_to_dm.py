from collections import defaultdict
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.polys.constructor import construct_domain
from sympy.polys.solvers import PolyNonlinearError
from .sdm import (
from sympy.utilities.misc import filldedent
def sympy_dict_to_dm(eqs_coeffs, eqs_rhs, syms):
    """Convert a system of dict equations to a sparse augmented matrix"""
    elems = set(eqs_rhs).union(*(e.values() for e in eqs_coeffs))
    K, elems_K = construct_domain(elems, field=True, extension=True)
    elem_map = dict(zip(elems, elems_K))
    neqs = len(eqs_coeffs)
    nsyms = len(syms)
    sym2index = dict(zip(syms, range(nsyms)))
    eqsdict = []
    for eq, rhs in zip(eqs_coeffs, eqs_rhs):
        eqdict = {sym2index[s]: elem_map[c] for s, c in eq.items()}
        if rhs:
            eqdict[nsyms] = -elem_map[rhs]
        if eqdict:
            eqsdict.append(eqdict)
    sdm_aug = SDM(enumerate(eqsdict), (neqs, nsyms + 1), K)
    return sdm_aug