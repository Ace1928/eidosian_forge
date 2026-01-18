from math import log10, floor
from ..units import html_of_unit, latex_of_unit, unicode_of_unit, to_unitless, unit_of
from ..util.parsing import _unicode_sup
def number_to_scientific_latex(number, uncertainty=None, unit=None, fmt=None):
    """Formats a number as LaTeX (optionally with unit/uncertainty)

    Parameters
    ----------
    number : float (w or w/o unit)
    uncertainty : same as number
    unit : unit
    fmt : int or callable

    Examples
    --------
    >>> number_to_scientific_latex(3.14) == '3.14'
    True
    >>> number_to_scientific_latex(3.14159265e-7)
    '3.1416\\\\cdot 10^{-7}'
    >>> import quantities as pq
    >>> number_to_scientific_latex(2**0.5 * pq.m / pq.s)
    '1.4142\\\\,\\\\mathrm{\\\\frac{m}{s}}'
    >>> number_to_scientific_latex(1.23456, .789, fmt=2)
    '1.23(79)'

    """
    return _number_to_X(number, uncertainty, unit, fmt, latex_of_unit, _latex_pow_10, '\\,')