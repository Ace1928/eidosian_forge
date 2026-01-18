from pyomo.common.deprecation import deprecation_warning
from pyomo.core.expr.numvalue import native_numeric_types
from pyomo.core.expr.numeric_expr import mutable_expression, NPV_SumExpression
from pyomo.core.base.var import Var
from pyomo.core.base.expression import Expression
from pyomo.core.base.component import _ComponentBase
import logging
def quicksum(args, start=0, linear=None):
    """A utility function to compute a sum of Pyomo expressions.

    The behavior of :func:`quicksum` is similar to the builtin
    :func:`sum` function, but this function can avoid the generation and
    disposal of intermediate objects, and thus is slightly more
    performant.

    Parameters
    ----------
    args: Iterable
        A generator for terms in the sum.

    start: Any
        A value that initializes the sum.  If this value is not a
        numeric constant, then the += operator is used to add terms to
        this object.  Defaults to 0.

    linear: bool
        DEPRECATED: the linearity of the resulting expression is
        determined automatically.  This option is ignored.

    Returns
    -------
    The value of the sum, which may be a Pyomo expression object.

    """
    try:
        args = iter(args)
    except:
        logger.error('The argument `args` to quicksum() is not iterable!')
        raise
    if linear is not None:
        deprecation_warning('The quicksum(linear=...) argument is deprecated and ignored.', version='6.6.0')
    if start.__class__ in native_numeric_types:
        with mutable_expression() as e:
            e += start
            for arg in args:
                e += arg
        if e.__class__ is NPV_SumExpression and all((arg.__class__ in native_numeric_types for arg in e.args)):
            return e()
        if e.nargs() > 1:
            return e
        elif not e.nargs():
            return 0
        else:
            return e.arg(0)
    e = start
    for arg in args:
        e += arg
    return e