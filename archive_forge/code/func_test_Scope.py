import math
from sympy.core.containers import Tuple
from sympy.core.numbers import nan, oo, Float, Integer
from sympy.core.relational import Lt
from sympy.core.symbol import symbols, Symbol
from sympy.functions.elementary.trigonometric import sin
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.testing.pytest import raises
from sympy.codegen.ast import (
def test_Scope():
    assign = Assignment(x, y)
    incr = AddAugmentedAssignment(x, 1)
    scp = Scope([assign, incr])
    cblk = CodeBlock(assign, incr)
    assert scp.body == cblk
    assert scp == Scope(cblk)
    assert scp != Scope([incr, assign])
    assert scp.func(*scp.args) == scp