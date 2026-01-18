from collections import defaultdict
from sympy.assumptions.ask import Q
from sympy.core import (Add, Mul, Pow, Number, NumberSymbol, Symbol)
from sympy.core.numbers import ImaginaryUnit
from sympy.functions.elementary.complexes import Abs
from sympy.logic.boolalg import (Equivalent, And, Or, Implies)
from sympy.matrices.expressions import MatMul
def multiregister(self, *classes):

    def _(func):
        for cls in classes:
            self.multifacts[cls] |= {func}
        return func
    return _