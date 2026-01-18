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
def test_issue_13516():
    if not matplotlib:
        skip('Matplotlib not the default backend')
    x = Symbol('x')
    pm = plot(sin(x), backend='matplotlib', show=False)
    assert pm.backend == MatplotlibBackend
    assert len(pm[0].get_data()[0]) >= 30
    pt = plot(sin(x), backend='text', show=False)
    assert pt.backend == TextBackend
    assert len(pt[0].get_data()[0]) >= 30
    pd = plot(sin(x), backend='default', show=False)
    assert pd.backend == DefaultBackend
    assert len(pd[0].get_data()[0]) >= 30
    p = plot(sin(x), show=False)
    assert p.backend == DefaultBackend
    assert len(p[0].get_data()[0]) >= 30