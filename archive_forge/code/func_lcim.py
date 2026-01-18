from .accumulationbounds import AccumBounds, AccumulationBounds # noqa: F401
from .singularities import singularities
from sympy.core import Pow, S
from sympy.core.function import diff, expand_mul
from sympy.core.kind import NumberKind
from sympy.core.mod import Mod
from sympy.core.numbers import equal_valued
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.polys.polytools import degree, lcm_list
from sympy.sets.sets import (Interval, Intersection, FiniteSet, Union,
from sympy.sets.fancysets import ImageSet
from sympy.utilities import filldedent
from sympy.utilities.iterables import iterable
def lcim(numbers):
    """Returns the least common integral multiple of a list of numbers.

    The numbers can be rational or irrational or a mixture of both.
    `None` is returned for incommensurable numbers.

    Parameters
    ==========

    numbers : list
        Numbers (rational and/or irrational) for which lcim is to be found.

    Returns
    =======

    number
        lcim if it exists, otherwise ``None`` for incommensurable numbers.

    Examples
    ========

    >>> from sympy.calculus.util import lcim
    >>> from sympy import S, pi
    >>> lcim([S(1)/2, S(3)/4, S(5)/6])
    15/2
    >>> lcim([2*pi, 3*pi, pi, pi/2])
    6*pi
    >>> lcim([S(1), 2*pi])
    """
    result = None
    if all((num.is_irrational for num in numbers)):
        factorized_nums = [num.factor() for num in numbers]
        factors_num = [num.as_coeff_Mul() for num in factorized_nums]
        term = factors_num[0][1]
        if all((factor == term for coeff, factor in factors_num)):
            common_term = term
            coeffs = [coeff for coeff, factor in factors_num]
            result = lcm_list(coeffs) * common_term
    elif all((num.is_rational for num in numbers)):
        result = lcm_list(numbers)
    else:
        pass
    return result