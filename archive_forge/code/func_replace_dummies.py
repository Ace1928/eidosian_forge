from sympy.concrete.summations import Sum
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import eye
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.hadamard import HadamardPower
from sympy.matrices.expressions.matexpr import (MatrixSymbol,
from sympy.matrices.expressions.matpow import MatPow
from sympy.matrices.expressions.special import (ZeroMatrix, Identity,
from sympy.matrices.expressions.trace import Trace, trace
from sympy.matrices.immutable import ImmutableMatrix
from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
from sympy.testing.pytest import XFAIL, raises
def replace_dummies(expr):
    repl = {i: Symbol(i.name) for i in expr.atoms(Dummy)}
    return expr.xreplace(repl)