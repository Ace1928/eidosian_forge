from .plot import BaseSeries, Plot
from .experimental_lambdify import experimental_lambdify, vectorized_lambdify
from .intervalmath import interval
from sympy.core.relational import (Equality, GreaterThan, LessThan,
from sympy.core.containers import Tuple
from sympy.core.relational import Eq
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.logic.boolalg import BooleanFunction
from sympy.polys.polyutils import _sort_gens
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import flatten
import warnings
def refine_pixels(interval_list):
    """ Evaluates the intervals and subdivides the interval if the
            expression is partially satisfied."""
    temp_interval_list = []
    plot_list = []
    for intervals in interval_list:
        intervalx = intervals[0]
        intervaly = intervals[1]
        func_eval = func(intervalx, intervaly)
        if func_eval[1] is False or func_eval[0] is False:
            pass
        elif func_eval == (True, True):
            plot_list.append([intervalx, intervaly])
        elif func_eval[1] is None or func_eval[0] is None:
            avgx = intervalx.mid
            avgy = intervaly.mid
            a = interval(intervalx.start, avgx)
            b = interval(avgx, intervalx.end)
            c = interval(intervaly.start, avgy)
            d = interval(avgy, intervaly.end)
            temp_interval_list.append([a, c])
            temp_interval_list.append([a, d])
            temp_interval_list.append([b, c])
            temp_interval_list.append([b, d])
    return (temp_interval_list, plot_list)