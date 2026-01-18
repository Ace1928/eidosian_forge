from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module
def test_compound_assignment_operator():
    c_src = 'void func()' + '{' + '\n' + 'int a = 100;' + '\n' + 'a += 10;' + '\n' + 'a -= 10;' + '\n' + 'a *= 10;' + '\n' + 'a /= 10;' + '\n' + 'a %= 10;' + '\n' + '}'
    res = SymPyExpression(c_src, 'c').return_expr()
    assert res[0] == FunctionDefinition(NoneToken(), name=String('func'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('a'), type=IntBaseType(String('intc')), value=Integer(100))), AddAugmentedAssignment(Variable(Symbol('a')), Integer(10)), SubAugmentedAssignment(Variable(Symbol('a')), Integer(10)), MulAugmentedAssignment(Variable(Symbol('a')), Integer(10)), DivAugmentedAssignment(Variable(Symbol('a')), Integer(10)), ModAugmentedAssignment(Variable(Symbol('a')), Integer(10))))