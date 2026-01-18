from sympy.assumptions.ask import Q
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.symbol import symbols
from sympy.logic.boolalg import (And, Or)
from sympy.assumptions.sathandlers import (ClassFactRegistry, allargs,
def test_class_handler_registry():
    my_handler_registry = ClassFactRegistry()

    @my_handler_registry.register(Mul)
    def fact1(expr):
        pass

    @my_handler_registry.multiregister(Expr)
    def fact2(expr):
        pass
    assert my_handler_registry[Basic] == (frozenset(), frozenset())
    assert my_handler_registry[Expr] == (frozenset(), frozenset({fact2}))
    assert my_handler_registry[Mul] == (frozenset({fact1}), frozenset({fact2}))