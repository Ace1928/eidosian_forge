import collections
from .visitor import StreamBasedExpressionVisitor
from .numvalue import nonpyomo_leaf_types
from pyomo.core.expr import (
from typing import List
from pyomo.common.collections import Sequence
from pyomo.common.errors import PyomoException
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import native_types
def assertExpressionsStructurallyEqual(test, a, b, include_named_exprs=True, places=None):
    """unittest-based assertion for comparing expressions

    This converts the expressions `a` and `b` into prefix notation and
    then compares the resulting lists.  Operators and (non-native type)
    leaf nodes in the prefix representation are converted to strings
    before comparing (so that things like variables can be compared
    across clones or pickles)

    Parameters
    ----------
    test: unittest.TestCase
        The unittest `TestCase` class that is performing the test.

    a: ExpressionBase or native type

    b: ExpressionBase or native type

    include_named_exprs: bool
       If True (the default), the comparison expands all named
       expressions when generating the prefix notation

    """
    prefix_a = convert_expression_to_prefix_notation(a, include_named_exprs)
    prefix_b = convert_expression_to_prefix_notation(b, include_named_exprs)
    for prefix in (prefix_a, prefix_b):
        for i, v in enumerate(prefix):
            if type(v) in native_types:
                continue
            if type(v) is tuple:
                if len(v) == 3:
                    prefix[i] = v[:2] + (str(v[2]),)
                continue
            prefix[i] = str(v)
    try:
        test.assertEqual(len(prefix_a), len(prefix_b))
        for _a, _b in zip(prefix_a, prefix_b):
            if _a.__class__ not in native_types and _b.__class__ not in native_types:
                test.assertIs(_a.__class__, _b.__class__)
            if places is None:
                test.assertEqual(_a, _b)
            else:
                test.assertAlmostEqual(_a, _b, places=places)
    except (PyomoException, AssertionError):
        test.fail(f'Expressions not structurally equal:\n\t{tostr(prefix_a)}\n\t!=\n\t{tostr(prefix_b)}')