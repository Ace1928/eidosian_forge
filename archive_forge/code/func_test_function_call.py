from sympy.parsing.sym_expr import SymPyExpression
from sympy.testing.pytest import raises, XFAIL
from sympy.external import import_module
def test_function_call():
    c_src1 = 'int fun1(int x)' + '\n' + '{' + '\n' + 'return x;' + '\n' + '}' + '\n' + 'void caller()' + '\n' + '{' + '\n' + 'int x = fun1(2);' + '\n' + '}'
    c_src2 = 'int fun2(int a, int b, int c)' + '\n' + '{' + '\n' + 'return a;' + '\n' + '}' + '\n' + 'void caller()' + '\n' + '{' + '\n' + 'int y = fun2(2, 3, 4);' + '\n' + '}'
    c_src3 = 'int fun3(int a, int b, int c)' + '\n' + '{' + '\n' + 'return b;' + '\n' + '}' + '\n' + 'void caller()' + '\n' + '{' + '\n' + 'int p;' + '\n' + 'int q;' + '\n' + 'int r;' + '\n' + 'int z = fun3(p, q, r);' + '\n' + '}'
    c_src4 = 'int fun4(float a, float b, int c)' + '\n' + '{' + '\n' + 'return c;' + '\n' + '}' + '\n' + 'void caller()' + '\n' + '{' + '\n' + 'float x;' + '\n' + 'float y;' + '\n' + 'int z;' + '\n' + 'int i = fun4(x, y, z)' + '\n' + '}'
    c_src5 = 'int fun()' + '\n' + '{' + '\n' + 'return 1;' + '\n' + '}' + '\n' + 'void caller()' + '\n' + '{' + '\n' + 'int a = fun()' + '\n' + '}'
    res1 = SymPyExpression(c_src1, 'c').return_expr()
    res2 = SymPyExpression(c_src2, 'c').return_expr()
    res3 = SymPyExpression(c_src3, 'c').return_expr()
    res4 = SymPyExpression(c_src4, 'c').return_expr()
    res5 = SymPyExpression(c_src5, 'c').return_expr()
    assert res1[0] == FunctionDefinition(IntBaseType(String('intc')), name=String('fun1'), parameters=(Variable(Symbol('x'), type=IntBaseType(String('intc'))),), body=CodeBlock(Return('x')))
    assert res1[1] == FunctionDefinition(NoneToken(), name=String('caller'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('x'), value=FunctionCall(String('fun1'), function_args=(Integer(2),))))))
    assert res2[0] == FunctionDefinition(IntBaseType(String('intc')), name=String('fun2'), parameters=(Variable(Symbol('a'), type=IntBaseType(String('intc'))), Variable(Symbol('b'), type=IntBaseType(String('intc'))), Variable(Symbol('c'), type=IntBaseType(String('intc')))), body=CodeBlock(Return('a')))
    assert res2[1] == FunctionDefinition(NoneToken(), name=String('caller'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('y'), value=FunctionCall(String('fun2'), function_args=(Integer(2), Integer(3), Integer(4)))))))
    assert res3[0] == FunctionDefinition(IntBaseType(String('intc')), name=String('fun3'), parameters=(Variable(Symbol('a'), type=IntBaseType(String('intc'))), Variable(Symbol('b'), type=IntBaseType(String('intc'))), Variable(Symbol('c'), type=IntBaseType(String('intc')))), body=CodeBlock(Return('b')))
    assert res3[1] == FunctionDefinition(NoneToken(), name=String('caller'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('p'), type=IntBaseType(String('intc')))), Declaration(Variable(Symbol('q'), type=IntBaseType(String('intc')))), Declaration(Variable(Symbol('r'), type=IntBaseType(String('intc')))), Declaration(Variable(Symbol('z'), value=FunctionCall(String('fun3'), function_args=(Symbol('p'), Symbol('q'), Symbol('r')))))))
    assert res4[0] == FunctionDefinition(IntBaseType(String('intc')), name=String('fun4'), parameters=(Variable(Symbol('a'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8))), Variable(Symbol('b'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8))), Variable(Symbol('c'), type=IntBaseType(String('intc')))), body=CodeBlock(Return('c')))
    assert res4[1] == FunctionDefinition(NoneToken(), name=String('caller'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('x'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)))), Declaration(Variable(Symbol('y'), type=FloatType(String('float32'), nbits=Integer(32), nmant=Integer(23), nexp=Integer(8)))), Declaration(Variable(Symbol('z'), type=IntBaseType(String('intc')))), Declaration(Variable(Symbol('i'), value=FunctionCall(String('fun4'), function_args=(Symbol('x'), Symbol('y'), Symbol('z')))))))
    assert res5[0] == FunctionDefinition(IntBaseType(String('intc')), name=String('fun'), parameters=(), body=CodeBlock(Return('')))
    assert res5[1] == FunctionDefinition(NoneToken(), name=String('caller'), parameters=(), body=CodeBlock(Declaration(Variable(Symbol('a'), value=FunctionCall(String('fun'), function_args=())))))