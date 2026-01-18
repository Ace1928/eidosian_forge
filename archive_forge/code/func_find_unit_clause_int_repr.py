from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Or, Not, conjuncts, disjuncts, to_cnf, \
from sympy.assumptions.cnf import CNF
from sympy.logic.inference import pl_true, literal_symbol
def find_unit_clause_int_repr(clauses, model):
    """
    Same as find_unit_clause, but arguments are expected to be in
    integer representation.

    >>> from sympy.logic.algorithms.dpll import find_unit_clause_int_repr
    >>> find_unit_clause_int_repr([{1, 2, 3},
    ...     {2, -3}, {1, -2}], {1: True})
    (2, False)

    """
    bound = set(model) | {-sym for sym in model}
    for clause in clauses:
        unbound = clause - bound
        if len(unbound) == 1:
            p = unbound.pop()
            if p < 0:
                return (-p, False)
            else:
                return (p, True)
    return (None, None)