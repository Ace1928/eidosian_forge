from sympy.utilities.exceptions import sympy_deprecation_warning
from .facts import FactRules, FactKB
from .sympify import sympify
from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions
def as_property(fact):
    """Convert a fact name to the name of the corresponding property"""
    return 'is_%s' % fact