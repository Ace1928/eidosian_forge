import collections
import enum
import logging
import math
import operator
from pyomo.common.dependencies import attempt_import
from pyomo.common.deprecation import deprecated, relocated_module_attribute
from pyomo.common.errors import PyomoException, DeveloperError
from pyomo.common.formatting import tostr
from pyomo.common.numeric_types import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.expr_common import (
from pyomo.core.expr.base import ExpressionBase, NPV_Mixin, visitor
def enable_expression_optimizations(zero=None, one=None):
    """Enable(disable) expression generation optimizations

    There are currently two optimizations available during expression generation:

    - zero: aggressively resolve `0*f(.)` expressions to `0`, `0/f(.)`
      expressions to `0`, and `f(.)**0` expressions to `1`

    - one: aggressively resolve identities: `1*f(.)` expressions to
      `f(.)`, `f(.)/1` expressions to `f(.)`, and `f(.)**1` expressions
      to `f(.)`.

    The default optimizations are `zero=False` and `one=True`.

    Notes
    -----

    Enabling the `zero` optimization can mask certain modeling errors.
    In particular, the optimization will suppress `ZeroDivisionError`s
    that should be raised if `f(.)` resolves to `0` (in the case of
    `0/f(.)`), as well as any errors that would have otherwise been
    raised during the evaluation of `f(.)`.  In addition, optimizing
    `f(.)**0 == 1` is only valid when `f(.)!=0`.  **Users who enable
    this optimization bear responsibility for ensuring that these
    optimizations will be valid for the model.**

    The `one` optimizations should generally be safe.

    Parameters
    ----------
    zero: bool, optional

        If `True` (`False`), enable (disable) the "zero" optimizations.
        If None, leave the optimization state unchanged.

    one: bool, optional

        If `True` (`False`), enable (disable) the "one" optimizations.
        If None, leave the optimization state unchanged.

    """
    for arg, key in ((zero, 0), (one, 1)):
        if arg is None:
            continue
        if arg:
            _zero_one_optimizations.add(key)
        else:
            _zero_one_optimizations.discard(key)