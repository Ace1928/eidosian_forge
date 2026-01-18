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
def test_ast_replace():
    x = Variable('x', real)
    y = Variable('y', real)
    n = Variable('n', integer)
    pwer = FunctionDefinition(real, 'pwer', [x, n], [pow(x.symbol, n.symbol)])
    pname = pwer.name
    pcall = FunctionCall('pwer', [y, 3])
    tree1 = CodeBlock(pwer, pcall)
    assert str(tree1.args[0].name) == 'pwer'
    assert str(tree1.args[1].name) == 'pwer'
    for a, b in zip(tree1, [pwer, pcall]):
        assert a == b
    tree2 = tree1.replace(pname, String('power'))
    assert str(tree1.args[0].name) == 'pwer'
    assert str(tree1.args[1].name) == 'pwer'
    assert str(tree2.args[0].name) == 'power'
    assert str(tree2.args[1].name) == 'power'