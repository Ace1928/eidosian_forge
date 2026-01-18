from sympy.combinatorics import Permutation as Perm
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core import Basic, Tuple, default_sort_key
from sympy.sets import FiniteSet
from sympy.utilities.iterables import (minlex, unflatten, flatten)
from sympy.utilities.misc import as_int
@property
def array_form(self):
    """Return the indices of the corners.

        The indices are given relative to the original position of corners.

        Examples
        ========

        >>> from sympy.combinatorics.polyhedron import tetrahedron
        >>> tetrahedron = tetrahedron.copy()
        >>> tetrahedron.array_form
        [0, 1, 2, 3]

        >>> tetrahedron.rotate(0)
        >>> tetrahedron.array_form
        [0, 2, 3, 1]
        >>> tetrahedron.pgroup[0].array_form
        [0, 2, 3, 1]

        See Also
        ========

        corners, cyclic_form
        """
    corners = list(self.args[0])
    return [corners.index(c) for c in self.corners]