from cmath import inf
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy import Matrix, pi
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import zeros
from sympy import sin, cos
def change_node_label(self, label, new_label):
    """
        This method changes the label of a node.

        Parameters
        ==========
        label: String or Symbol
            The label of the node for which the label has
            to be changed.

        new_label: String or Symbol
            The new label of the node.

        Examples
        ========

        >>> from sympy.physics.continuum_mechanics.truss import Truss
        >>> t = Truss()
        >>> t.add_node('A', 0, 0)
        >>> t.add_node('B', 3, 0)
        >>> t.nodes
        [('A', 0, 0), ('B', 3, 0)]
        >>> t.change_node_label('A', 'C')
        >>> t.nodes
        [('C', 0, 0), ('B', 3, 0)]
        """
    if label not in self._node_labels:
        raise ValueError('No such node exists for the Truss')
    elif new_label in self._node_labels:
        raise ValueError('A node with the given label already exists')
    else:
        for node in self._nodes:
            if node[0] == label:
                self._nodes[self._nodes.index((label, node[1], node[2]))] = (new_label, node[1], node[2])
                self._node_labels[self._node_labels.index(node[0])] = new_label
                self._node_coordinates[new_label] = self._node_coordinates[label]
                self._node_coordinates.pop(label)
                if node[0] in list(self._supports):
                    self._supports[new_label] = self._supports[node[0]]
                    self._supports.pop(node[0])
                if new_label in list(self._supports):
                    if self._supports[new_label] == 'pinned':
                        if 'R_' + str(label) + '_x' in list(self._reaction_loads) and 'R_' + str(label) + '_y' in list(self._reaction_loads):
                            self._reaction_loads['R_' + str(new_label) + '_x'] = self._reaction_loads['R_' + str(label) + '_x']
                            self._reaction_loads['R_' + str(new_label) + '_y'] = self._reaction_loads['R_' + str(label) + '_y']
                            self._reaction_loads.pop('R_' + str(label) + '_x')
                            self._reaction_loads.pop('R_' + str(label) + '_y')
                        self._loads[new_label] = self._loads[label]
                        for load in self._loads[new_label]:
                            if load[1] == 90:
                                load[0] -= Symbol('R_' + str(label) + '_y')
                                if load[0] == 0:
                                    self._loads[label].remove(load)
                                break
                        for load in self._loads[new_label]:
                            if load[1] == 0:
                                load[0] -= Symbol('R_' + str(label) + '_x')
                                if load[0] == 0:
                                    self._loads[label].remove(load)
                                break
                        self.apply_load(new_label, Symbol('R_' + str(new_label) + '_x'), 0)
                        self.apply_load(new_label, Symbol('R_' + str(new_label) + '_y'), 90)
                        self._loads.pop(label)
                    elif self._supports[new_label] == 'roller':
                        self._loads[new_label] = self._loads[label]
                        for load in self._loads[label]:
                            if load[1] == 90:
                                load[0] -= Symbol('R_' + str(label) + '_y')
                                if load[0] == 0:
                                    self._loads[label].remove(load)
                                break
                        self.apply_load(new_label, Symbol('R_' + str(new_label) + '_y'), 90)
                        self._loads.pop(label)
                elif label in list(self._loads):
                    self._loads[new_label] = self._loads[label]
                    self._loads.pop(label)
                for member in list(self._members):
                    if self._members[member][0] == node[0]:
                        self._members[member][0] = new_label
                        self._nodes_occupied[new_label, self._members[member][1]] = True
                        self._nodes_occupied[self._members[member][1], new_label] = True
                        self._nodes_occupied.pop((label, self._members[member][1]))
                        self._nodes_occupied.pop((self._members[member][1], label))
                    elif self._members[member][1] == node[0]:
                        self._members[member][1] = new_label
                        self._nodes_occupied[self._members[member][0], new_label] = True
                        self._nodes_occupied[new_label, self._members[member][0]] = True
                        self._nodes_occupied.pop((self._members[member][0], label))
                        self._nodes_occupied.pop((label, self._members[member][0]))