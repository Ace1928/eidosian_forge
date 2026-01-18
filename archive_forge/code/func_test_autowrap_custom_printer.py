import sympy
import tempfile
import os
from sympy.core.mod import Mod
from sympy.core.relational import Eq
from sympy.core.symbol import symbols
from sympy.external import import_module
from sympy.tensor import IndexedBase, Idx
from sympy.utilities.autowrap import autowrap, ufuncify, CodeWrapError
from sympy.testing.pytest import skip
def test_autowrap_custom_printer():
    has_module('Cython')
    from sympy.core.numbers import pi
    from sympy.utilities.codegen import C99CodeGen
    from sympy.printing.c import C99CodePrinter

    class PiPrinter(C99CodePrinter):

        def _print_Pi(self, expr):
            return 'S_PI'
    printer = PiPrinter()
    gen = C99CodeGen(printer=printer)
    gen.preprocessor_statements.append('#include "shortpi.h"')
    expr = pi * a
    expected = '#include "%s"\n#include <math.h>\n#include "shortpi.h"\n\ndouble autofunc(double a) {\n\n   double autofunc_result;\n   autofunc_result = S_PI*a;\n   return autofunc_result;\n\n}\n'
    tmpdir = tempfile.mkdtemp()
    with open(os.path.join(tmpdir, 'shortpi.h'), 'w') as f:
        f.write('#define S_PI 3.14')
    func = autowrap(expr, backend='cython', tempdir=tmpdir, code_gen=gen)
    assert func(4.2) == 3.14 * 4.2
    for filename in os.listdir(tmpdir):
        if filename.startswith('wrapped_code') and filename.endswith('.c'):
            with open(os.path.join(tmpdir, filename)) as f:
                lines = f.readlines()
                expected = expected % filename.replace('.c', '.h')
                assert ''.join(lines[7:]) == expected