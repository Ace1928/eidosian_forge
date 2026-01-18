from cmath import inf
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
from sympy import sin, cos
def add_member(self, label, start, end):
    """
        This method adds a member between any two nodes in the given truss.

        Parameters
        ==========
        label: String or Symbol
            The label for a member. It is the only way to identify a particular member.

        start: String or Symbol
            The label of the starting point/node of the member.

        end: String or Symbol
            The label of the ending point/node of the member.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.add_node('B', 3, 0)
        >>> t.add_node('C', 2, 2)
        >>> t.add_member('AB', 'A', 'B')
        >>> t.members
        {'AB': ['A', 'B']}
        """
    if start not in self._node_labels or end not in self._node_labels or start == end:
        raise ValueError('The start and end points of the member must be unique nodes')
    elif label in list(self._members):
        raise ValueError('A member with the same label already exists for the truss')
    elif self._nodes_occupied.get((start, end)):
        raise ValueError('A member already exists between the two nodes')
    else:
        self._members[label] = [start, end]
        self._nodes_occupied[start, end] = True
        self._nodes_occupied[end, start] = True
        self._internal_forces[label] = 0