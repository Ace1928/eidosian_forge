import os
import tempfile
from sympy.core.symbol import (Symbol, symbols)
from sympy.codegen.ast import (
from sympy.codegen.fnodes import (
from sympy.codegen.futils import render_as_module
from sympy.core.expr import unchanged
from sympy.external import import_module
from sympy.printing.codeprinter import fcode
from sympy.utilities._compilation import has_fortran, compile_run_strings, compile_link_import_strings
from sympy.utilities._compilation.util import may_xfail
from sympy.testing.pytest import skip, XFAIL
@may_xfail
def test_bind_C():
    if not has_fortran():
        skip('No fortran compiler found.')
    if not cython:
        skip('Cython not found.')
    if not np:
        skip('NumPy not found.')
    a = Symbol('a', real=True)
    s = Symbol('s', integer=True)
    body = [Return((sum_(a ** 2) / s) ** 0.5)]
    arr = array(a, dim=[s], intent='in')
    fd = FunctionDefinition(real, 'rms', [arr, s], body, attrs=[bind_C('rms')])
    f_mod = render_as_module([fd], 'mod_rms')
    with tempfile.TemporaryDirectory() as folder:
        mod, info = compile_link_import_strings([('rms.f90', f_mod), ('_rms.pyx', '#cython: language_level={}\n'.format('3') + 'cdef extern double rms(double*, int*)\ndef py_rms(double[::1] x):\n    cdef int s = x.size\n    return rms(&x[0], &s)\n')], build_dir=folder)
        assert abs(mod.py_rms(np.array([2.0, 4.0, 2.0, 2.0])) - 7 ** 0.5) < 1e-14