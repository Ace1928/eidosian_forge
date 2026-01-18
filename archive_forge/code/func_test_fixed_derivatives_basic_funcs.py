from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import zip
from builtins import map
from builtins import range
import copy
import weakref
import math
from math import isnan, isinf
import random
import sys
import uncertainties.core as uncert_core
from uncertainties.core import ufloat, AffineScalarFunc, ufloat_fromstr
from uncertainties import umath
def test_fixed_derivatives_basic_funcs():
    """
    Pre-calculated derivatives for operations on AffineScalarFunc.
    """

    def check_op(op, num_args):
        """
        Makes sure that the derivatives for function '__op__' of class
        AffineScalarFunc, which takes num_args arguments, are correct.

        If num_args is None, a correct value is calculated.
        """
        op_string = '__%s__' % op
        func = getattr(AffineScalarFunc, op_string)
        numerical_derivatives = uncert_core.NumericalDerivatives(lambda *args: func(*map(uncert_core.to_affine_scalar, args)))
        compare_derivatives(func, numerical_derivatives, [num_args])
    for op in uncert_core.modified_operators:
        check_op(op, 1)
    for op in uncert_core.modified_ops_with_reflection:
        check_op(op, 2)