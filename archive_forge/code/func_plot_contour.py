from collections.abc import Callable
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import arity, Function
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.external import import_module
from sympy.printing.latex import latex
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from .experimental_lambdify import (vectorized_lambdify, lambdify)
from sympy.plotting.textplot import textplot
def plot_contour(*args, show=True, **kwargs):
    """
    Draws contour plot of a function

    Usage
    =====

    Single plot

    ``plot_contour(expr, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plot with the same range.

    ``plot_contour(expr1, expr2, range_x, range_y, **kwargs)``

    If the ranges are not specified, then a default range of (-10, 10) is used.

    Multiple plots with different ranges.

    ``plot_contour((expr1, range_x, range_y), (expr2, range_x, range_y), ..., **kwargs)``

    Ranges have to be specified for every expression.

    Default range may change in the future if a more advanced default range
    detection algorithm is implemented.

    Arguments
    =========

    expr : Expression representing the function along x.

    range_x : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the x variable, e.g. (x, 0, 5).

    range_y : (:class:`Symbol`, float, float)
        A 3-tuple denoting the range of the y variable, e.g. (y, 0, 5).

    Keyword Arguments
    =================

    Arguments for ``ContourSeries`` class:

    nb_of_points_x : int
        The x range is sampled uniformly at ``nb_of_points_x`` of points.

    nb_of_points_y : int
        The y range is sampled uniformly at ``nb_of_points_y`` of points.

    Aesthetics:

    surface_color : Function which returns a float
        Specifies the color for the surface of the plot. See
        :class:`sympy.plotting.Plot` for more details.

    If there are multiple plots, then the same series arguments are applied to
    all the plots. If you want to set these options separately, you can index
    the returned ``Plot`` object and set it.

    Arguments for ``Plot`` class:

    title : str
        Title of the plot.

    size : (float, float), optional
        A tuple in the form (width, height) in inches to specify the size of
        the overall figure. The default value is set to ``None``, meaning
        the size will be set by the default backend.

    See Also
    ========

    Plot, ContourSeries

    """
    args = list(map(sympify, args))
    plot_expr = check_arguments(args, 1, 2)
    series = [ContourSeries(*arg) for arg in plot_expr]
    plot_contours = Plot(*series, **kwargs)
    if len(plot_expr[0].free_symbols) > 2:
        raise ValueError('Contour Plot cannot Plot for more than two variables.')
    if show:
        plot_contours.show()
    return plot_contours