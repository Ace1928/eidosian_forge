from sympy.core import (pi, symbols, Rational, Integer, GoldenRatio, EulerGamma,
from sympy.functions import Piecewise, sin, cos, Abs, exp, ceiling, sqrt
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.printing.glsl import GLSLPrinter
from sympy.printing.str import StrPrinter
from sympy.utilities.lambdify import implemented_function
from sympy.tensor import IndexedBase, Idx
from sympy.matrices import Matrix, MatrixSymbol
from sympy.core import Tuple
from sympy.printing.glsl import glsl_code
import textwrap
def test_MxN_mats():
    generatedAssertions = 'def test_misc_mats():\n'
    for i in range(1, 6):
        for j in range(1, 6):
            A = Matrix([[x + y * j for x in range(j)] for y in range(i)])
            gl = glsl_code(A)
            glTransposed = glsl_code(A, mat_transpose=True)
            generatedAssertions += '    mat = ' + StrPrinter()._print(A) + '\n\n'
            generatedAssertions += "    gl = '''" + gl + "'''\n"
            generatedAssertions += "    glTransposed = '''" + glTransposed + "'''\n\n"
            generatedAssertions += '    assert glsl_code(mat) == gl\n'
            generatedAssertions += '    assert glsl_code(mat,mat_transpose=True) == glTransposed\n'
            if i == 1 and j == 1:
                assert gl == '0'
            elif i <= 4 and j <= 4 and (i > 1) and (j > 1):
                assert gl.startswith('mat%s' % j)
                assert glTransposed.startswith('mat%s' % i)
            elif i == 1 and j <= 4:
                assert gl.startswith('vec')
            elif j == 1 and i <= 4:
                assert gl.startswith('vec')
            elif i == 1:
                assert gl.startswith('float[%s](' % j * i)
                assert glTransposed.startswith('float[%s](' % j * i)
            elif j == 1:
                assert gl.startswith('float[%s](' % i * j)
                assert glTransposed.startswith('float[%s](' % i * j)
            else:
                assert gl.startswith('float[%s](' % (i * j))
                assert glTransposed.startswith('float[%s](' % (i * j))
                glNested = glsl_code(A, mat_nested=True)
                glNestedTransposed = glsl_code(A, mat_transpose=True, mat_nested=True)
                assert glNested.startswith('float[%s][%s]' % (i, j))
                assert glNestedTransposed.startswith('float[%s][%s]' % (j, i))
                generatedAssertions += "    glNested = '''" + glNested + "'''\n"
                generatedAssertions += "    glNestedTransposed = '''" + glNestedTransposed + "'''\n\n"
                generatedAssertions += '    assert glsl_code(mat,mat_nested=True) == glNested\n'
                generatedAssertions += '    assert glsl_code(mat,mat_nested=True,mat_transpose=True) == glNestedTransposed\n\n'
    generateAssertions = False
    if generateAssertions:
        gen = open('test_glsl_generated_matrices.py', 'w')
        gen.write(generatedAssertions)
        gen.close()