from sympy.categories import (CompositeMorphism, IdentityMorphism,
from sympy.core import Dict, Symbol, default_sort_key
from sympy.printing.latex import latex
from sympy.sets import FiniteSet
from sympy.utilities.iterables import iterable
from sympy.utilities.decorator import doctest_depends_on
from itertools import chain
def set_label_position(free1, free2, pos1, pos2, backwards, m_str_info):
    """
            Given the information about room available to one side and
            to the other side of a morphism (``free1`` and ``free2``),
            sets the position of the morphism label in such a way that
            it is on the freer side.  This latter operations involves
            choice between ``pos1`` and ``pos2``, taking ``backwards``
            in consideration.

            Thus this function will do nothing if either both ``free1
            == True`` and ``free2 == True`` or both ``free1 == False``
            and ``free2 == False``.  In either case, choosing one side
            over the other presents no advantage.
            """
    if backwards:
        pos1, pos2 = (pos2, pos1)
    if free1 and (not free2):
        m_str_info.label_position = pos1
    elif free2 and (not free1):
        m_str_info.label_position = pos2