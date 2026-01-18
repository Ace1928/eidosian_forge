from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def group_size(group):
    """
            For the supplied group (or object, eventually), returns
            the size of the cell that will hold this group (object).
            """
    if group in groups_grids:
        grid = groups_grids[group]
        return (grid.height, grid.width)
    else:
        return (1, 1)