from sympy.assumptions.assume import (global_assumptions, Predicate,
from sympy.assumptions.cnf import CNF, EncodedCNF, Literal
from sympy.core import sympify
from sympy.core.kind import BooleanKind
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.logic.inference import satisfiable
from sympy.utilities.decorator import memoize_property
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.assumptions.ask_generated import (get_all_known_facts,
@memoize_property
def extended_nonnegative(self):
    from .handlers.order import ExtendedNonNegativePredicate
    return ExtendedNonNegativePredicate()