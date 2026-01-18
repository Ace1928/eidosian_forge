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
@XFAIL
@may_xfail
def test_Subroutine():
    r = Symbol('r', real=True)
    i = Symbol('i', integer=True)
    v_r = Variable.deduced(r, attrs=(dimension(assumed_extent), intent_out))
    v_i = Variable.deduced(i)
    v_n = Variable('n', integer)
    do_loop = Do([Assignment(Element(r, [i]), literal_dp(1) / i ** 2)], i, 1, v_n)
    sub = Subroutine('f', [v_r], [Declaration(v_n), Declaration(v_i), Assignment(v_n, size(r)), do_loop])
    x = Symbol('x', real=True)
    v_x3 = Variable.deduced(x, attrs=[dimension(3)])
    mod = Module('mymod', definitions=[sub])
    prog = Program('foo', [use(mod, only=[sub]), Declaration(v_x3), SubroutineCall(sub, [v_x3]), Print([sum_(v_x3), v_x3])])
    if not has_fortran():
        skip('No fortran compiler found.')
    (stdout, stderr), info = compile_run_strings([('a.f90', fcode(mod, standard=90)), ('b.f90', fcode(prog, standard=90))], clean=True)
    ref = [1.0 / i ** 2 for i in range(1, 4)]
    assert str(sum(ref))[:-3] in stdout
    for _ in ref:
        assert str(_)[:-3] in stdout
    assert stderr == ''