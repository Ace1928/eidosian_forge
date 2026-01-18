from ..sage_helper import _within_sage
from . import exhaust
from .links_base import Link
def jones_polynomial(link, normalized=True):
    bracket = kauffman_bracket(link)
    if normalized:
        factor = q + q ** (-1)
        norm_bracket = bracket // factor
        assert norm_bracket * factor == bracket
    else:
        norm_bracket = bracket
    signs = [c.sign for c in link.crossings]
    n_minus, n_plus = (signs.count(-1), signs.count(1))
    return (-1) ** n_minus * q ** (n_plus - 2 * n_minus) * norm_bracket