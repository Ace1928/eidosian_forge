from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Or, Not, conjuncts, disjuncts, to_cnf, \
from sympy.assumptions.cnf import CNF
from sympy.logic.inference import pl_true, literal_symbol
def dpll(clauses, symbols, model):
    """
    Compute satisfiability in a partial model.
    Clauses is an array of conjuncts.

    >>> from sympy.abc import A, B, D
    >>> from sympy.logic.algorithms.dpll import dpll
    >>> dpll([A, B, D], [A, B], {D: False})
    False

    """
    P, value = find_unit_clause(clauses, model)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = ~P
        clauses = unit_propagate(clauses, P)
        P, value = find_unit_clause(clauses, model)
    P, value = find_pure_symbol(symbols, clauses)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = ~P
        clauses = unit_propagate(clauses, P)
        P, value = find_pure_symbol(symbols, clauses)
    unknown_clauses = []
    for c in clauses:
        val = pl_true(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    if not clauses:
        return model
    P = symbols.pop()
    model_copy = model.copy()
    model.update({P: True})
    model_copy.update({P: False})
    symbols_copy = symbols[:]
    return dpll(unit_propagate(unknown_clauses, P), symbols, model) or dpll(unit_propagate(unknown_clauses, Not(P)), symbols_copy, model_copy)