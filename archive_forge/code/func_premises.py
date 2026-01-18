from sympy.core import S, Basic, Dict, Symbol, Tuple, sympify
from sympy.core.symbol import Str
from sympy.sets import Set, FiniteSet, EmptySet
from sympy.utilities.iterables import iterable
@property
def premises(self):
    """
        Returns the premises of this diagram.

        Examples
        ========

        >>> from sympy.categories import Object, NamedMorphism
        >>> from sympy.categories import IdentityMorphism, Diagram
        >>> from sympy import pretty
        >>> A = Object("A")
        >>> B = Object("B")
        >>> f = NamedMorphism(A, B, "f")
        >>> id_A = IdentityMorphism(A)
        >>> id_B = IdentityMorphism(B)
        >>> d = Diagram([f])
        >>> print(pretty(d.premises, use_unicode=False))
        {id:A-->A: EmptySet, id:B-->B: EmptySet, f:A-->B: EmptySet}

        """
    return self.args[0]