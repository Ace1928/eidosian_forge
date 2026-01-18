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
def test_all_comparison_ops(x, y):
    """
        Takes two Variable objects.

        Fails if any comparison operation fails to follow the proper
        semantics: a comparison only returns True if the correspond float
        comparison results are True for all the float values taken by
        the variables (of x and y) when they vary in an infinitesimal
        neighborhood within their uncertainty.

        This test is stochastic: it may, exceptionally, fail for
        correctly implemented comparison operators.
        """
    import random

    def random_float(var):
        """
            Returns a random value for Variable var, in an
            infinitesimal interval withing its uncertainty.  The case
            of a zero uncertainty is special.
            """
        return (random.random() - 0.5) * min(var.std_dev, 1e-05) + var.nominal_value
    for op in ['__%s__' % name for name in ('ne', 'eq', 'lt', 'le', 'gt', 'ge')]:
        try:
            float_func = getattr(float, op)
        except AttributeError:
            continue
        sampled_results = []
        sampled_results.append(float_func(x.nominal_value, y.nominal_value))
        for check_num in range(50):
            sampled_results.append(float_func(random_float(x), random_float(y)))
        min_result = min(sampled_results)
        max_result = max(sampled_results)
        if min_result == max_result:
            correct_result = min_result
        else:
            num_min_result = sampled_results.count(min_result)
            correct_result = num_min_result == 1
        try:
            assert correct_result == getattr(x, op)(y)
        except AssertionError:
            print('Sampling results:', sampled_results)
            raise Exception('Semantic value of %s %s (%s) %s not correctly reproduced.' % (x, op, y, correct_result))