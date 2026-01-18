from sympy.combinatorics.free_groups import free_group
from sympy.printing.defaults import DefaultPrinting
from itertools import chain, product
from bisect import bisect_left
def coset_enumeration_c(fp_grp, Y, max_cosets=None, draft=None, incomplete=False):
    """
    >>> from sympy.combinatorics.free_groups import free_group
    >>> from sympy.combinatorics.fp_groups import FpGroup, coset_enumeration_c
    >>> F, x, y = free_group("x, y")
    >>> f = FpGroup(F, [x**3, y**3, x**-1*y**-1*x*y])
    >>> C = coset_enumeration_c(f, [x])
    >>> C.table
    [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1]]

    """
    X = fp_grp.generators
    R = fp_grp.relators
    C = CosetTable(fp_grp, Y, max_cosets=max_cosets)
    if draft:
        C.table = draft.table[:]
        C.p = draft.p[:]
        C.deduction_stack = draft.deduction_stack
        for alpha, x in product(range(len(C.table)), X):
            if C.table[alpha][C.A_dict[x]] is not None:
                C.deduction_stack.append((alpha, x))
    A = C.A
    R_cyc_red = [rel.identity_cyclic_reduction() for rel in R]
    R_c = list(chain.from_iterable(((rel.cyclic_conjugates(), (rel ** (-1)).cyclic_conjugates()) for rel in R_cyc_red)))
    R_set = set()
    for conjugate in R_c:
        R_set = R_set.union(conjugate)
    R_c_list = []
    for x in C.A:
        r = {word for word in R_set if word[0] == x}
        R_c_list.append(r)
        R_set.difference_update(r)
    for w in Y:
        C.scan_and_fill_c(0, w)
    for x in A:
        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
    alpha = 0
    while alpha < len(C.table):
        if C.p[alpha] == alpha:
            try:
                for x in C.A:
                    if C.p[alpha] != alpha:
                        break
                    if C.table[alpha][C.A_dict[x]] is None:
                        C.define_c(alpha, x)
                        C.process_deductions(R_c_list[C.A_dict[x]], R_c_list[C.A_dict_inv[x]])
            except ValueError as e:
                if incomplete:
                    return C
                raise e
        alpha += 1
    return C