from .cartan_type import CartanType
from sympy.core.basic import Atom
def simple_roots(self):
    """Generate the simple roots of the Lie algebra

        The rank of the Lie algebra determines the number of simple roots that
        it has.  This method obtains the rank of the Lie algebra, and then uses
        the simple_root method from the Lie algebra classes to generate all the
        simple roots.

        Examples
        ========

        >>> from sympy.liealgebras.root_system import RootSystem
        >>> c = RootSystem("A3")
        >>> roots = c.simple_roots()
        >>> roots
        {1: [1, -1, 0, 0], 2: [0, 1, -1, 0], 3: [0, 0, 1, -1]}

        """
    n = self.cartan_type.rank()
    roots = {}
    for i in range(1, n + 1):
        root = self.cartan_type.simple_root(i)
        roots[i] = root
    return roots