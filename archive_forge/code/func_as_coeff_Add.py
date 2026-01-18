from typing import Tuple as tTuple
from collections import defaultdict
from functools import cmp_to_key, reduce
from operator import attrgetter
from .basic import Basic
from .parameters import global_parameters
from .logic import _fuzzy_group, fuzzy_or, fuzzy_not
from .singleton import S
from .operations import AssocOp, AssocOpDispatcher
from .cache import cacheit
from .numbers import ilcm, igcd, equal_valued
from .expr import Expr
from .kind import UndefinedKind
from sympy.utilities.iterables import is_sequence, sift
from .mul import Mul, _keep_coeff, _unevaluated_Mul
from .numbers import Rational
def as_coeff_Add(self, rational=False, deps=None):
    """
        Efficiently extract the coefficient of a summation.
        """
    coeff, args = (self.args[0], self.args[1:])
    if coeff.is_Number and (not rational) or coeff.is_Rational:
        return (coeff, self._new_rawargs(*args))
    return (S.Zero, self)