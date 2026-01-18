from sympy.testing.pytest import raises
from sympy.parsing.sym_expr import SymPyExpression
from sympy.external import import_module
def test_mul_binop():
    src1 = src + '            d = a + b - c\n            c = a * b + d\n            s = p * q / r\n            r = p * s + q / p\n            '
    expr1.convert_to_expr(src1, 'f')
    ls1 = expr1.return_expr()
    for iter in range(8, 12):
        assert isinstance(ls1[iter], Assignment)
    assert ls1[8] == Assignment(Variable(Symbol('d')), Symbol('a') + Symbol('b') - Symbol('c'))
    assert ls1[9] == Assignment(Variable(Symbol('c')), Symbol('a') * Symbol('b') + Symbol('d'))
    assert ls1[10] == Assignment(Variable(Symbol('s')), Symbol('p') * Symbol('q') / Symbol('r'))
    assert ls1[11] == Assignment(Variable(Symbol('r')), Symbol('p') * Symbol('s') + Symbol('q') / Symbol('p'))