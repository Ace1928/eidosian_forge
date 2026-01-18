from sympy.combinatorics import Permutation as Perm
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core import Basic, Tuple, default_sort_key
from sympy.sets import FiniteSet
from sympy.utilities.iterables import (minlex, unflatten, flatten)
from sympy.utilities.misc import as_int
@property
def faces(self):
    """
        Get the faces of the Polyhedron.
        """
    return self._faces