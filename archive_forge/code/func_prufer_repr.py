from sympy.core import Basic
from sympy.core.containers import Tuple
from sympy.tensor.array import Array
from sympy.core.sympify import _sympify
from sympy.utilities.iterables import flatten, iterable
from sympy.utilities.misc import as_int
from collections import defaultdict
@property
def prufer_repr(self):
    """Returns Prufer sequence for the Prufer object.

        This sequence is found by removing the highest numbered vertex,
        recording the node it was attached to, and continuing until only
        two vertices remain. The Prufer sequence is the list of recorded nodes.

        Examples
        ========

        >>> from sympy.combinatorics.prufer import Prufer
        >>> Prufer([[0, 3], [1, 3], [2, 3], [3, 4], [4, 5]]).prufer_repr
        [3, 3, 3, 4]
        >>> Prufer([1, 0, 0]).prufer_repr
        [1, 0, 0]

        See Also
        ========

        to_prufer

        """
    if self._prufer_repr is None:
        self._prufer_repr = self.to_prufer(self._tree_repr[:], self.nodes)
    return self._prufer_repr