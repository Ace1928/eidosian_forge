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
def test_bode_data(sys):
    return y_coordinate_equality(bode_magnitude_numerical_data, bode_mag_evalf, sys) and y_coordinate_equality(bode_phase_numerical_data, bode_phase_evalf, sys)