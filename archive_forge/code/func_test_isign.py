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
def test_isign():
    x = Symbol('x', integer=True)
    assert unchanged(isign, 1, x)
    assert fcode(isign(1, x), standard=95, source_format='free') == 'isign(1, x)'