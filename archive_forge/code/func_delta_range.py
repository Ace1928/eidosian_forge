from math import prod
from sympy.core import S, Integer
from sympy.core.function import Function
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.iterables import has_dups
@property
def delta_range(self):
    if len(self.args) > 2:
        return self.args[2]