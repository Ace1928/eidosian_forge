from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def prepend_column(self):
    """
        Prepends the grid with an empty column.
        """
    self._width += 1
    for i in range(self._height):
        self._array[i].insert(0, None)