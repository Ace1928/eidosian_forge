from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises
from sympy.external import import_module
def test_fortran_parse():
    expr = SymPyExpression(src, 'f')
    ls = expr.return_expr()
    assert ls[0] == Declaration(Variable(Symbol('a'), type=IntBaseType(String('integer')), value=Integer(0)))
    assert ls[1] == Declaration(Variable(Symbol('b'), type=IntBaseType(String('integer')), value=Integer(0)))
    assert ls[2] == Declaration(Variable(Symbol('c'), type=IntBaseType(String('integer')), value=Integer(0)))
    assert ls[3] == Declaration(Variable(Symbol('d'), type=IntBaseType(String('integer')), value=Integer(0)))
    assert ls[4] == Declaration(Variable(Symbol('p'), type=FloatBaseType(String('real')), value=Float('0.0', precision=53)))
    assert ls[5] == Declaration(Variable(Symbol('q'), type=FloatBaseType(String('real')), value=Float('0.0', precision=53)))
    assert ls[6] == Declaration(Variable(Symbol('r'), type=FloatBaseType(String('real')), value=Float('0.0', precision=53)))
    assert ls[7] == Declaration(Variable(Symbol('s'), type=FloatBaseType(String('real')), value=Float('0.0', precision=53)))