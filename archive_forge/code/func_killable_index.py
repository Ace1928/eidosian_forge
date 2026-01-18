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
def killable_index(self):
    """
        Returns the index which is preferred to substitute in the final
        expression.

        Explanation
        ===========

        The index to substitute is the index with less information regarding
        fermi level. If indices contain the same information, 'a' is preferred
        before 'b'.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> j = Symbol('j', below_fermi=True)
        >>> p = Symbol('p')
        >>> KroneckerDelta(p, i).killable_index
        p
        >>> KroneckerDelta(p, a).killable_index
        p
        >>> KroneckerDelta(i, j).killable_index
        j

        See Also
        ========

        preferred_index

        """
    if self._get_preferred_index():
        return self.args[0]
    else:
        return self.args[1]