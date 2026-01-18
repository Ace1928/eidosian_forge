from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Or, Not, conjuncts, disjuncts, to_cnf, \
from sympy.assumptions.cnf import CNF
from sympy.logic.inference import pl_true, literal_symbol
def dpll_int_repr(clauses, symbols, model):
    """
    Compute satisfiability in a partial model.
    Arguments are expected to be in integer representation

    >>> from sympy.logic.algorithms.dpll import dpll_int_repr
    >>> dpll_int_repr([{1}, {2}, {3}], {1, 2}, {3: False})
    False

    """
    P, value = find_unit_clause_int_repr(clauses, model)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = -P
        clauses = unit_propagate_int_repr(clauses, P)
        P, value = find_unit_clause_int_repr(clauses, model)
    P, value = find_pure_symbol_int_repr(symbols, clauses)
    while P:
        model.update({P: value})
        symbols.remove(P)
        if not value:
            P = -P
        clauses = unit_propagate_int_repr(clauses, P)
        P, value = find_pure_symbol_int_repr(symbols, clauses)
    unknown_clauses = []
    for c in clauses:
        val = pl_true_int_repr(c, model)
        if val is False:
            return False
        if val is not True:
            unknown_clauses.append(c)
    if not unknown_clauses:
        return model
    P = symbols.pop()
    model_copy = model.copy()
    model.update({P: True})
    model_copy.update({P: False})
    symbols_copy = symbols.copy()
    return dpll_int_repr(unit_propagate_int_repr(unknown_clauses, P), symbols, model) or dpll_int_repr(unit_propagate_int_repr(unknown_clauses, -P), symbols_copy, model_copy)