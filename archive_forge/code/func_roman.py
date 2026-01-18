from math import log10, floor
from ..units import html_of_unit, latex_of_unit, unicode_of_unit, to_unitless, unit_of
from ..util.parsing import _unicode_sup
def roman(num):
    """
    Examples
    --------
    >>> roman(4)
    'IV'
    >>> roman(17)
    'XVII'

    """
    tokens = 'M CM D CD C XC L XL X IX V IV I'.split()
    values = (1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1)
    result = ''
    for t, v in zip(tokens, values):
        cnt = num // v
        result += t * cnt
        num -= v * cnt
    return result