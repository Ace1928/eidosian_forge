from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def prepend_row(self):
    """
        Prepends the grid with an empty row.
        """
    self._height += 1
    self._array.insert(0, [None for j in range(self._width)])