import sympy
from sympy.parsing.sympy_parser import (
from sympy.testing.pytest import raises
def test_implicit_multiplication():
    cases = {'5x': '5*x', 'abc': 'a*b*c', '3sin(x)': '3*sin(x)', '(x+1)(x+2)': '(x+1)*(x+2)', '(5 x**2)sin(x)': '(5*x**2)*sin(x)', '2 sin(x) cos(x)': '2*sin(x)*cos(x)', 'pi x': 'pi*x', 'x pi': 'x*pi', 'E x': 'E*x', 'EulerGamma y': 'EulerGamma*y', 'E pi': 'E*pi', 'pi (x + 2)': 'pi*(x+2)', '(x + 2) pi': '(x+2)*pi', 'pi sin(x)': 'pi*sin(x)'}
    transformations = standard_transformations + (convert_xor,)
    transformations2 = transformations + (split_symbols, implicit_multiplication)
    for case in cases:
        implicit = parse_expr(case, transformations=transformations2)
        normal = parse_expr(cases[case], transformations=transformations)
        assert implicit == normal
    application = ['sin x', 'cos 2*x', 'sin cos x']
    for case in application:
        raises(SyntaxError, lambda: parse_expr(case, transformations=transformations2))
    raises(TypeError, lambda: parse_expr('sin**2(x)', transformations=transformations2))