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
def test_FunctionPrototype_and_FunctionDefinition():
    vx = Variable(x, type=real)
    vn = Variable(n, type=integer)
    fp1 = FunctionPrototype(real, 'power', [vx, vn])
    assert fp1.return_type == real
    assert fp1.name == String('power')
    assert fp1.parameters == Tuple(vx, vn)
    assert fp1 == FunctionPrototype(real, 'power', [vx, vn])
    assert fp1 != FunctionPrototype(real, 'power', [vn, vx])
    assert fp1.func(*fp1.args) == fp1
    body = [Assignment(x, x ** n), Return(x)]
    fd1 = FunctionDefinition(real, 'power', [vx, vn], body)
    assert fd1.return_type == real
    assert str(fd1.name) == 'power'
    assert fd1.parameters == Tuple(vx, vn)
    assert fd1.body == CodeBlock(*body)
    assert fd1 == FunctionDefinition(real, 'power', [vx, vn], body)
    assert fd1 != FunctionDefinition(real, 'power', [vx, vn], body[::-1])
    assert fd1.func(*fd1.args) == fd1
    fp2 = FunctionPrototype.from_FunctionDefinition(fd1)
    assert fp2 == fp1
    fd2 = FunctionDefinition.from_FunctionPrototype(fp1, body)
    assert fd2 == fd1