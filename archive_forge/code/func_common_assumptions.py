from sympy.utilities.exceptions import sympy_deprecation_warning
from .facts import FactRules, FactKB
from .sympify import sympify
from sympy.core.random import _assumptions_shuffle as shuffle
from sympy.core.assumptions_generated import generated_assumptions as _assumptions
def common_assumptions(exprs, check=None):
    """return those assumptions which have the same True or False
    value for all the given expressions.

    Examples
    ========

    >>> from sympy.core import common_assumptions
    >>> from sympy import oo, pi, sqrt
    >>> common_assumptions([-4, 0, sqrt(2), 2, pi, oo])
    {'commutative': True, 'composite': False,
    'extended_real': True, 'imaginary': False, 'odd': False}

    By default, all assumptions are tested; pass an iterable of the
    assumptions to limit those that are reported:

    >>> common_assumptions([0, 1, 2], ['positive', 'integer'])
    {'integer': True}
    """
    check = _assume_defined if check is None else set(check)
    if not check or not exprs:
        return {}
    assume = [assumptions(i, _check=check) for i in sympify(exprs)]
    for i, e in enumerate(assume):
        assume[i] = {k: e[k] for k in set(e) & check}
    common = set.intersection(*[set(i) for i in assume])
    a = assume[0]
    return {k: a[k] for k in common if all((a[k] == b[k] for b in assume))}