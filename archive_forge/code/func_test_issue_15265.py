import os
from tempfile import TemporaryDirectory
from sympy.concrete.summations import Sum
from sympy.core.numbers import (I, oo, pi)
from sympy.core.relational import Ne
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (LambertW, exp, exp_polar, log)
from sympy.functions.elementary.miscellaneous import (real_root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import And
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.plotting.plot import (
from sympy.plotting.plot import (
from sympy.testing.pytest import skip, raises, warns, warns_deprecated_sympy
from sympy.utilities import lambdify as lambdify_
from sympy.utilities.exceptions import ignore_warnings
def test_issue_15265():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    eqn = sin(x)
    p = plot(eqn, xlim=(-S.Pi, S.Pi), ylim=(-1, 1))
    p._backend.close()
    p = plot(eqn, xlim=(-1, 1), ylim=(-S.Pi, S.Pi))
    p._backend.close()
    p = plot(eqn, xlim=(-1, 1), ylim=(sympify('-3.14'), sympify('3.14')))
    p._backend.close()
    p = plot(eqn, xlim=(sympify('-3.14'), sympify('3.14')), ylim=(-1, 1))
    p._backend.close()
    raises(ValueError, lambda: plot(eqn, xlim=(-S.ImaginaryUnit, 1), ylim=(-1, 1)))
    raises(ValueError, lambda: plot(eqn, xlim=(-1, 1), ylim=(-1, S.ImaginaryUnit)))
    raises(ValueError, lambda: plot(eqn, xlim=(S.NegativeInfinity, 1), ylim=(-1, 1)))
    raises(ValueError, lambda: plot(eqn, xlim=(-1, 1), ylim=(-1, S.Infinity)))