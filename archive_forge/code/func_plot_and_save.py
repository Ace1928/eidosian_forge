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
def plot_and_save(expr, *args, name='', dir=None, **kwargs):
    p = plot_implicit(expr, *args, **kwargs)
    p.save(tmp_file(dir=dir, name=name))
    p._backend.close()