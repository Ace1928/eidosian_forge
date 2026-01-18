from cmath import inf
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
from sympy import sin, cos
def apply_support(self, location, type):
    """
        This method adds a pinned or roller support at a particular node

        Parameters
        ==========

        location: String or Symbol
            Label of the Node at which support is added.

        type: String
            Type of the support being provided at the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.add_node('B', 3, 0)
        >>> t.apply_support('A', 'pinned')
        >>> t.supports
        {'A': 'pinned'}
        """
    if location not in self._node_labels:
        raise ValueError('Support must be added on a known node')
    else:
        if location not in list(self._supports):
            if type == 'pinned':
                self.apply_load(location, Symbol('R_' + str(location) + '_x'), 0)
                self.apply_load(location, Symbol('R_' + str(location) + '_y'), 90)
            elif type == 'roller':
                self.apply_load(location, Symbol('R_' + str(location) + '_y'), 90)
        elif self._supports[location] == 'pinned':
            if type == 'roller':
                self.remove_load(location, Symbol('R_' + str(location) + '_x'), 0)
        elif self._supports[location] == 'roller':
            if type == 'pinned':
                self.apply_load(location, Symbol('R_' + str(location) + '_x'), 0)
        self._supports[location] = type