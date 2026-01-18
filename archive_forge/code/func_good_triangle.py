from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def good_triangle(tri):
    objs = DiagramGrid._triangle_objects(tri)
    return obj in objs and placed_objects & objs - {obj} == set()