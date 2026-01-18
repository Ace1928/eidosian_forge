from sympy.printing.dot import (purestr, styleof, attrprint, dotnode,
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import (Float, Integer)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing.repr import srepr
from sympy.abc import x
def test_Matrix_and_non_basics():
    from sympy.matrices.expressions.matexpr import MatrixSymbol
    n = Symbol('n')
    assert dotprint(MatrixSymbol('X', n, n)) == 'digraph{\n\n# Graph style\n"ordering"="out"\n"rankdir"="TD"\n\n#########\n# Nodes #\n#########\n\n"MatrixSymbol(Str(\'X\'), Symbol(\'n\'), Symbol(\'n\'))_()" ["color"="black", "label"="MatrixSymbol", "shape"="ellipse"];\n"Str(\'X\')_(0,)" ["color"="blue", "label"="X", "shape"="ellipse"];\n"Symbol(\'n\')_(1,)" ["color"="black", "label"="n", "shape"="ellipse"];\n"Symbol(\'n\')_(2,)" ["color"="black", "label"="n", "shape"="ellipse"];\n\n#########\n# Edges #\n#########\n\n"MatrixSymbol(Str(\'X\'), Symbol(\'n\'), Symbol(\'n\'))_()" -> "Str(\'X\')_(0,)";\n"MatrixSymbol(Str(\'X\'), Symbol(\'n\'), Symbol(\'n\'))_()" -> "Symbol(\'n\')_(1,)";\n"MatrixSymbol(Str(\'X\'), Symbol(\'n\'), Symbol(\'n\'))_()" -> "Symbol(\'n\')_(2,)";\n}'