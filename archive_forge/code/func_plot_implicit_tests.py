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
def plot_implicit_tests(name):
    temp_dir = mkdtemp()
    TmpFileManager.tmp_folder(temp_dir)
    x = Symbol('x')
    y = Symbol('y')
    plot_and_save(Eq(y, cos(x)), (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(Eq(y ** 2, x ** 3 - x), (x, -5, 5), (y, -4, 4), name=name, dir=temp_dir)
    plot_and_save(y > 1 / x, (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y < 1 / tan(x), (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y >= 2 * sin(x) * cos(x), (x, -5, 5), (y, -2, 2), name=name, dir=temp_dir)
    plot_and_save(y <= x ** 2, (x, -3, 3), (y, -1, 5), name=name, dir=temp_dir)
    plot_and_save(Eq(y ** 2, x ** 3 - x), dir=temp_dir)
    plot_and_save(Eq(y ** 2, x ** 3 - x), adaptive=False, dir=temp_dir)
    plot_and_save(Eq(y ** 2, x ** 3 - x), adaptive=False, points=500, dir=temp_dir)
    plot_and_save(y > x, (x, -5, 5), dir=temp_dir)
    plot_and_save(And(y > exp(x), y > x + 2), dir=temp_dir)
    plot_and_save(Or(y > x, y > -x), dir=temp_dir)
    plot_and_save(x ** 2 - 1, (x, -5, 5), dir=temp_dir)
    plot_and_save(x ** 2 - 1, dir=temp_dir)
    plot_and_save(y > x, depth=-5, dir=temp_dir)
    plot_and_save(y > x, depth=5, dir=temp_dir)
    plot_and_save(y > cos(x), adaptive=False, dir=temp_dir)
    plot_and_save(y < cos(x), adaptive=False, dir=temp_dir)
    plot_and_save(And(y > cos(x), Or(y > x, Eq(y, x))), dir=temp_dir)
    plot_and_save(y - cos(pi / x), dir=temp_dir)
    plot_and_save(x ** 2 - 1, title='An implicit plot', dir=temp_dir)