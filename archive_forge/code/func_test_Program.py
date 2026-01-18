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
def test_Program():
    x = Symbol('x', real=True)
    vx = Variable.deduced(x, 42)
    decl = Declaration(vx)
    prnt = Print([x, x + 1])
    prog = Program('foo', [decl, prnt])
    if not has_fortran():
        skip('No fortran compiler found.')
    (stdout, stderr), info = compile_run_strings([('main.f90', fcode(prog, standard=90))], clean=True)
    assert '42' in stdout
    assert '43' in stdout
    assert stderr == ''
    assert info['exit_status'] == os.EX_OK