from sympy.core.numbers import I, pi
from sympy.functions.elementary.exponential import (exp, log)
from sympy.polys.partfrac import apart
from sympy.core.symbol import Dummy
from sympy.external import import_module
from sympy.functions import arg, Abs
from sympy.integrals.laplace import _fast_inverse_laplace
from sympy.physics.control.lti import SISOLinearTimeInvariant
from sympy.plotting.plot import LineOver1DRangeSeries
from sympy.polys.polytools import Poly
from sympy.printing.latex import latex
def ramp_response_plot(system, slope=1, color='b', prec=8, lower_limit=0, upper_limit=10, show_axes=False, grid=True, show=True, **kwargs):
    """
    Returns the ramp response of a continuous-time system.

    Ramp function is defined as the straight line
    passing through origin ($f(x) = mx$). The slope of
    the ramp function can be varied by the user and
    the default value is 1.

    Parameters
    ==========

    system : SISOLinearTimeInvariant type
        The LTI SISO system for which the Ramp Response is to be computed.
    slope : Number, optional
        The slope of the input ramp function. Defaults to 1.
    color : str, tuple, optional
        The color of the line. Default is Blue.
    show : boolean, optional
        If ``True``, the plot will be displayed otherwise
        the equivalent matplotlib ``plot`` object will be returned.
        Defaults to True.
    lower_limit : Number, optional
        The lower limit of the plot range. Defaults to 0.
    upper_limit : Number, optional
        The upper limit of the plot range. Defaults to 10.
    prec : int, optional
        The decimal point precision for the point coordinate values.
        Defaults to 8.
    show_axes : boolean, optional
        If ``True``, the coordinate axes will be shown. Defaults to False.
    grid : boolean, optional
        If ``True``, the plot will have a grid. Defaults to True.

    Examples
    ========

    .. plot::
        :context: close-figs
        :format: doctest
        :include-source: True

        >>> from sympy.abc import s
        >>> from sympy.physics.control.lti import TransferFunction
        >>> from sympy.physics.control.control_plots import ramp_response_plot
        >>> tf1 = TransferFunction(s, (s+4)*(s+8), s)
        >>> ramp_response_plot(tf1, upper_limit=2)   # doctest: +SKIP

    See Also
    ========

    step_response_plot, ramp_response_plot

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Ramp_function

    """
    x, y = ramp_response_numerical_data(system, slope=slope, prec=prec, lower_limit=lower_limit, upper_limit=upper_limit, **kwargs)
    plt.plot(x, y, color=color)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Ramp Response of ${latex(system)}$ [Slope = {slope}]', pad=20)
    if grid:
        plt.grid()
    if show_axes:
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
    if show:
        plt.show()
        return
    return plt