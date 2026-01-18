from sympy.assumptions.ask import Q
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (to_cnf, And, Not, Implies, Equivalent,
from sympy.logic.inference import satisfiable
@cacheit
def get_composite_predicates():
    return {Q.real: Q.negative | Q.zero | Q.positive, Q.integer: Q.even | Q.odd, Q.nonpositive: Q.negative | Q.zero, Q.nonzero: Q.negative | Q.positive, Q.nonnegative: Q.zero | Q.positive, Q.extended_real: Q.negative_infinite | Q.negative | Q.zero | Q.positive | Q.positive_infinite, Q.extended_positive: Q.positive | Q.positive_infinite, Q.extended_negative: Q.negative | Q.negative_infinite, Q.extended_nonzero: Q.negative_infinite | Q.negative | Q.positive | Q.positive_infinite, Q.extended_nonpositive: Q.negative_infinite | Q.negative | Q.zero, Q.extended_nonnegative: Q.zero | Q.positive | Q.positive_infinite, Q.complex: Q.algebraic | Q.transcendental}