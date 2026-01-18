import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
@staticmethod
def from_sqrt_lin_combination(l):
    """
        Construct from a SqrtLinCombination.
        """

    def to_set(k):
        if k == _One:
            return frozenset()
        else:
            return frozenset([k])
    return _FactorizedSqrtLinCombination(dict(((to_set(k), v) for k, v in l._dict.items())), embed_cache=l._embed_cache)