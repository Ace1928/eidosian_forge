from sympy.assumptions.ask import Q
from sympy.assumptions.assume import AppliedPredicate
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import (to_cnf, And, Not, Implies, Equivalent,
from sympy.logic.inference import satisfiable
@cacheit
def get_known_facts_keys():
    """
    Return every unary predicates registered to ``Q``.

    This function is used to generate the keys for
    ``generate_known_facts_dict``.

    """
    exclude = set()
    for pred in [Q.eq, Q.ne, Q.gt, Q.lt, Q.ge, Q.le]:
        exclude.add(pred)
    result = []
    for attr in Q.__class__.__dict__:
        if attr.startswith('__'):
            continue
        pred = getattr(Q, attr)
        if pred in exclude:
            continue
        result.append(pred)
    return result