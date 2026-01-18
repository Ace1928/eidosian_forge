from __future__ import division
from __future__ import absolute_import
import sys
import math
from uncertainties import ufloat
import uncertainties.core as uncert_core
import uncertainties.umath_core as umath_core
from . import test_uncertainties
def test_fixed_derivatives_math_funcs():
    """
    Comparison between function derivatives and numerical derivatives.

    This comparison is useful for derivatives that are analytical.
    """
    for name in umath_core.many_scalars_to_scalar_funcs:
        func = getattr(umath_core, name)
        numerical_derivatives = uncert_core.NumericalDerivatives(lambda *args: func(*args))
        test_uncertainties.compare_derivatives(func, numerical_derivatives)

    def frac_part_modf(x):
        return umath_core.modf(x)[0]

    def int_part_modf(x):
        return umath_core.modf(x)[1]
    test_uncertainties.compare_derivatives(frac_part_modf, uncert_core.NumericalDerivatives(lambda x: frac_part_modf(x)))
    test_uncertainties.compare_derivatives(int_part_modf, uncert_core.NumericalDerivatives(lambda x: int_part_modf(x)))

    def mantissa_frexp(x):
        return umath_core.frexp(x)[0]

    def exponent_frexp(x):
        return umath_core.frexp(x)[1]
    test_uncertainties.compare_derivatives(mantissa_frexp, uncert_core.NumericalDerivatives(lambda x: mantissa_frexp(x)))
    test_uncertainties.compare_derivatives(exponent_frexp, uncert_core.NumericalDerivatives(lambda x: exponent_frexp(x)))