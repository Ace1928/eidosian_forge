from sympy.core.numbers import (I, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.logic.boolalg import (And, Or)
from sympy.plotting.plot_implicit import plot_implicit
from sympy.plotting.plot import unset_show
from tempfile import NamedTemporaryFile, mkdtemp
from sympy.testing.pytest import skip, warns, XFAIL
from sympy.external import import_module
from sympy.testing.tmpfiles import TmpFileManager
import os
@XFAIL
def test_no_adaptive_meshing():
    matplotlib = import_module('matplotlib', min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        try:
            temp_dir = mkdtemp()
            TmpFileManager.tmp_folder(temp_dir)
            x = Symbol('x')
            y = Symbol('y')
            with warns(UserWarning, match='Adaptive meshing could not be applied'):
                plot_and_save(Eq(y, re(cos(x) + I * sin(x))), name='test', dir=temp_dir)
        finally:
            TmpFileManager.cleanup()
    else:
        skip('Matplotlib not the default backend')