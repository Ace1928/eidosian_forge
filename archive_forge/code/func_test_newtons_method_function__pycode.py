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
def test_newtons_method_function__pycode():
    x = sp.Symbol('x', real=True)
    expr = sp.cos(x) - x ** 3
    func = newtons_method_function(expr, x)
    py_mod = py_module(func)
    namespace = {}
    exec(py_mod, namespace, namespace)
    res = eval('newton(0.5)', namespace)
    assert abs(res - 0.865474033102) < 1e-12