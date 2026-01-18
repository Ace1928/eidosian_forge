from sympy.core.sorting import default_sort_key
from sympy.logic.boolalg import Or, Not, conjuncts, disjuncts, to_cnf, \
from sympy.assumptions.cnf import CNF
from sympy.logic.inference import pl_true, literal_symbol
def find_pure_symbol_int_repr(symbols, unknown_clauses):
    """
    Same as find_pure_symbol, but arguments are expected
    to be in integer representation

    >>> from sympy.logic.algorithms.dpll import find_pure_symbol_int_repr
    >>> find_pure_symbol_int_repr({1,2,3},
    ...     [{1, -2}, {-2, -3}, {3, 1}])
    (1, True)

    """
    all_symbols = set().union(*unknown_clauses)
    found_pos = all_symbols.intersection(symbols)
    found_neg = all_symbols.intersection([-s for s in symbols])
    for p in found_pos:
        if -p not in found_neg:
            return (p, True)
    for p in found_neg:
        if -p not in found_pos:
            return (-p, False)
    return (None, None)