from sympy.core import S, Basic, Dict, Symbol, Tuple, sympify
from sympy.core.symbol import Str
from sympy.sets import Set, FiniteSet, EmptySet
from sympy.utilities.iterables import iterable
def is_subdiagram(self, diagram):
    """
        Checks whether ``diagram`` is a subdiagram of ``self``.
        Diagram `D'` is a subdiagram of `D` if all premises
        (conclusions) of `D'` are contained in the premises
        (conclusions) of `D`.  The morphisms contained
        both in `D'` and `D` should have the same properties for `D'`
        to be a subdiagram of `D`.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism, Diagram
        >>> A = Object("A")
        >>> B = Object("B")
        >>> C = Object("C")
        >>> f = NamedMorphism(A, B, "f")
        >>> g = NamedMorphism(B, C, "g")
        >>> d = Diagram([f, g], {g * f: "unique"})
        >>> d1 = Diagram([f])
        >>> d.is_subdiagram(d1)
        True
        >>> d1.is_subdiagram(d)
        False
        """
    premises = all((m in self.premises and diagram.premises[m] == self.premises[m] for m in diagram.premises))
    if not premises:
        return False
    conclusions = all((m in self.conclusions and diagram.conclusions[m] == self.conclusions[m] for m in diagram.conclusions))
    return conclusions