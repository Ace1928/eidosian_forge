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
def test_logplot_PR_16796():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    p = plot(x, (x, 0.001, 100), xscale='log', show=False)
    assert len(p[0].get_data()[0]) >= 30
    assert p[0].end == 100.0
    assert p[0].start == 0.001