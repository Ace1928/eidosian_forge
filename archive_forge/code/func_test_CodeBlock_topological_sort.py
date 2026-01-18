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
def test_CodeBlock_topological_sort():
    assignments = [Assignment(x, y + z), Assignment(z, 1), Assignment(t, x), Assignment(y, 2)]
    ordered_assignments = [Assignment(z, 1), Assignment(y, 2), Assignment(x, y + z), Assignment(t, x)]
    c1 = CodeBlock.topological_sort(assignments)
    assert c1 == CodeBlock(*ordered_assignments)
    invalid_assignments = [Assignment(x, y + z), Assignment(z, 1), Assignment(y, x), Assignment(y, 2)]
    raises(ValueError, lambda: CodeBlock.topological_sort(invalid_assignments))
    free_assignments = [Assignment(x, y + z), Assignment(z, a * b), Assignment(t, x), Assignment(y, b + 3)]
    free_assignments_ordered = [Assignment(z, a * b), Assignment(y, b + 3), Assignment(x, y + z), Assignment(t, x)]
    c2 = CodeBlock.topological_sort(free_assignments)
    assert c2 == CodeBlock(*free_assignments_ordered)