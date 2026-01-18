import collections
from .visitor import StreamBasedExpressionVisitor
from .numvalue import nonpyomo_leaf_types
from pyomo.core.expr import (
from typing import List
from pyomo.common.collections import Sequence
from pyomo.common.errors import PyomoException
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import native_types
def assertExpressionsEqual(test, a, b, include_named_exprs=True, places=None):
    """unittest-based assertion for comparing expressions

    This converts the expressions `a` and `b` into prefix notation and
    then compares the resulting lists.

    Parameters
    ----------
    test: unittest.TestCase
        The unittest `TestCase` class that is performing the test.

    a: ExpressionBase or native type

    b: ExpressionBase or native type

    include_named_exprs: bool
       If True (the default), the comparison expands all named
       expressions when generating the prefix notation

    places: Number of decimal places required for equality of floating
            point numbers in the expression. If None (the default), the
            expressions must be exactly equal.
    """
    prefix_a = convert_expression_to_prefix_notation(a, include_named_exprs)
    prefix_b = convert_expression_to_prefix_notation(b, include_named_exprs)
    try:
        test.assertEqual(len(prefix_a), len(prefix_b))
        for _a, _b in zip(prefix_a, prefix_b):
            test.assertIs(_a.__class__, _b.__class__)
            if places is None:
                test.assertEqual(_a, _b)
            else:
                test.assertAlmostEqual(_a, _b, places=places)
    except (PyomoException, AssertionError):
        test.fail(f'Expressions not equal:\n\t{tostr(prefix_a)}\n\t!=\n\t{tostr(prefix_b)}')