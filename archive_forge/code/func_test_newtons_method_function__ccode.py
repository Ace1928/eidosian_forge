import tempfile
import sympy as sp
from sympy.codegen.ast import Assignment
from sympy.codegen.algorithms import newtons_method, newtons_method_function
from sympy.codegen.fnodes import bind_C
from sympy.codegen.futils import render_as_module as f_module
from sympy.codegen.pyutils import render_as_module as py_module
from sympy.external import import_module
from sympy.printing.codeprinter import ccode
from sympy.utilities._compilation import compile_link_import_strings, has_c, has_fortran
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, raises
@may_xfail
def test_newtons_method_function__ccode():
    x = sp.Symbol('x', real=True)
    expr = sp.cos(x) - x ** 3
    func = newtons_method_function(expr, x)
    if not cython:
        skip('cython not installed.')
    if not has_c():
        skip('No C compiler found.')
    compile_kw = {'std': 'c99'}
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([('newton.c', '#include <math.h>\n#include <stdio.h>\n' + ccode(func)), ('_newton.pyx', '#cython: language_level={}\n'.format('3') + 'cdef extern double newton(double)\ndef py_newton(x):\n    return newton(x)\n')], build_dir=folder, compile_kwargs=compile_kw)
        assert abs(mod.py_newton(0.5) - 0.865474033102) < 1e-12