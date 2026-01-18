import math
from typing import List
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.transpiler.exceptions import CouplingError
def add_physical_qubit(self, physical_qubit):
    """Add a physical qubit to the coupling graph as a node.

        physical_qubit (int): An integer representing a physical qubit.

        Raises:
            CouplingError: if trying to add duplicate qubit
        """
    if not isinstance(physical_qubit, int):
        raise CouplingError('Physical qubits should be integers.')
    if physical_qubit in self.physical_qubits:
        raise CouplingError('The physical qubit %s is already in the coupling graph' % physical_qubit)
    self.graph.add_node(physical_qubit)
    self._dist_matrix = None
    self._qubit_list = None
    self._size = None