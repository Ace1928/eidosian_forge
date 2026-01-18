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
def pole_zero_numerical_data(system):
    """
    Returns the numerical data of poles and zeros of the system.
    It is internally used by ``pole_zero_plot`` to get the data
    for plotting poles and zeros. Users can use this data to further
    analyse the dynamics of the system or plot using a different
    backend/plotting-module.

    Parameters
    ==========

    system : SISOLinearTimeInvariant
        The system for which the pole-zero data is to be computed.

    Returns
    =======

    tuple : (zeros, poles)
        zeros = Zeros of the system. NumPy array of complex numbers.
        poles = Poles of the system. NumPy array of complex numbers.

    Raises
    ======

    NotImplementedError
        When a SISO LTI system is not passed.

        When time delay terms are present in the system.

    ValueError
        When more than one free symbol is present in the system.
        The only variable in the transfer function should be
        the variable of the Laplace transform.

    Examples
    ========

    >>> from sympy.abc import s
    >>> from sympy.physics.control.lti import TransferFunction
    >>> from sympy.physics.control.control_plots import pole_zero_numerical_data
    >>> tf1 = TransferFunction(s**2 + 1, s**4 + 4*s**3 + 6*s**2 + 5*s + 2, s)
    >>> pole_zero_numerical_data(tf1)   # doctest: +SKIP
    ([-0.+1.j  0.-1.j], [-2. +0.j        -0.5+0.8660254j -0.5-0.8660254j -1. +0.j       ])

    See Also
    ========

    pole_zero_plot

    """
    _check_system(system)
    system = system.doit()
    num_poly = Poly(system.num, system.var).all_coeffs()
    den_poly = Poly(system.den, system.var).all_coeffs()
    num_poly = np.array(num_poly, dtype=np.complex128)
    den_poly = np.array(den_poly, dtype=np.complex128)
    zeros = np.roots(num_poly)
    poles = np.roots(den_poly)
    return (zeros, poles)