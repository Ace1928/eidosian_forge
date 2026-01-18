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
def pz_tester(sys, expected_value):
    z, p = pole_zero_numerical_data(sys)
    z_check = numpy.allclose(z, expected_value[0])
    p_check = numpy.allclose(p, expected_value[1])
    return p_check and z_check