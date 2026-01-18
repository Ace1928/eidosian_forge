from sympy.assumptions.ask import Q
from sympy.assumptions.wrapper import (AssumptionsWrapper, is_infinite,
from sympy.core.symbol import Symbol
from sympy.core.assumptions import _assume_defined
def test_all_predicates():
    for fact in _assume_defined:
        method_name = f'_eval_is_{fact}'
        assert hasattr(AssumptionsWrapper, method_name)