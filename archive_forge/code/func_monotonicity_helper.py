from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import sec, csc, cot, tan, cos
from sympy.utilities.misc import filldedent
def monotonicity_helper(expression, predicate, interval=S.Reals, symbol=None):
    """
    Helper function for functions checking function monotonicity.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked
    predicate : function
        The property being tested for. The function takes in an integer
        and returns a boolean. The integer input is the derivative and
        the boolean result should be true if the property is being held,
        and false otherwise.
    interval : Set, optional
        The range of values in which we are testing, defaults to all reals.
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    It returns a boolean indicating whether the interval in which
    the function's derivative satisfies given predicate is a superset
    of the given interval.

    Returns
    =======

    Boolean
        True if ``predicate`` is true for all the derivatives when ``symbol``
        is varied in ``range``, False otherwise.

    """
    from sympy.solvers.solveset import solveset
    expression = sympify(expression)
    free = expression.free_symbols
    if symbol is None:
        if len(free) > 1:
            raise NotImplementedError('The function has not yet been implemented for all multivariate expressions.')
    variable = symbol or (free.pop() if free else Symbol('x'))
    derivative = expression.diff(variable)
    predicate_interval = solveset(predicate(derivative), variable, S.Reals)
    return interval.is_subset(predicate_interval)