from cmath import inf
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
from sympy import sin, cos
def remove_support(self, location):
    """
        This method removes support from a particular node

        Parameters
        ==========

        location: String or Symbol
            Label of the Node at which support is to be removed.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.add_node('B', 3, 0)
        >>> t.apply_support('A', 'pinned')
        >>> t.supports
        {'A': 'pinned'}
        >>> t.remove_support('A')
        >>> t.supports
        {}
        """
    if location not in self._node_labels:
        raise ValueError('No such node exists in the Truss')
    elif location not in list(self._supports):
        raise ValueError('No support has been added to the given node')
    else:
        if self._supports[location] == 'pinned':
            self.remove_load(location, Symbol('R_' + str(location) + '_x'), 0)
            self.remove_load(location, Symbol('R_' + str(location) + '_y'), 90)
        elif self._supports[location] == 'roller':
            self.remove_load(location, Symbol('R_' + str(location) + '_y'), 90)
        self._supports.pop(location)