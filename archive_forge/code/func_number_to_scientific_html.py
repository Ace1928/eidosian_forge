from math import log10, floor
from ..units import html_of_unit, latex_of_unit, unicode_of_unit, to_unitless, unit_of
from ..util.parsing import _unicode_sup
def number_to_scientific_html(number, uncertainty=None, unit=None, fmt=None):
    """Formats a number as HTML (optionally with unit/uncertainty)

    Parameters
    ----------
    number : float (w or w/o unit)
    uncertainty : same as number
    unit : unit
    fmt : int or callable

    Examples
    --------
    >>> number_to_scientific_html(3.14) == '3.14'
    True
    >>> number_to_scientific_html(3.14159265e-7)
    '3.1416&sdot;10<sup>-7</sup>'
    >>> number_to_scientific_html(1e13)
    '10<sup>13</sup>'
    >>> import quantities as pq
    >>> number_to_scientific_html(2**0.5 * pq.m / pq.s)
    '1.4142 m/s'

    """
    return _number_to_X(number, uncertainty, unit, fmt, html_of_unit, _html_pow_10)