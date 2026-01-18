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
def indices_contain_equal_information(self):
    """
        Returns True if indices are either both above or below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, q).indices_contain_equal_information
        True
        >>> KroneckerDelta(p, q+1).indices_contain_equal_information
        True
        >>> KroneckerDelta(i, p).indices_contain_equal_information
        False

        """
    if self.args[0].assumptions0.get('below_fermi') and self.args[1].assumptions0.get('below_fermi'):
        return True
    if self.args[0].assumptions0.get('above_fermi') and self.args[1].assumptions0.get('above_fermi'):
        return True
    return self.is_below_fermi and self.is_above_fermi