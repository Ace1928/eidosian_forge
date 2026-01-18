from math import isclose
from sympy.core.numbers import I
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import (Abs, arg)
from sympy.functions.elementary.exponential import log
from sympy.abc import s, p, a
from sympy.external import import_module
from sympy.physics.control.control_plots import \
from sympy.physics.control.lti import (TransferFunction,
from sympy.testing.pytest import raises, skip
def bode_phase_evalf(system, point):
    expr = system.to_expr()
    _w = Dummy('w', real=True)
    w_expr = expr.subs({system.var: I * _w})
    return arg(w_expr).subs({_w: point}).evalf()