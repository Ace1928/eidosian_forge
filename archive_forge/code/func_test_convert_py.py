from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises
from sympy.external import import_module
def test_convert_py():
    src1 = src + '            a = b + c\n            s = p * q / r\n            '
    expr1.convert_to_expr(src1, 'f')
    exp_py = expr1.convert_to_python()
    assert exp_py == ['a = 0', 'b = 0', 'c = 0', 'd = 0', 'p = 0.0', 'q = 0.0', 'r = 0.0', 's = 0.0', 'a = b + c', 's = p*q/r']