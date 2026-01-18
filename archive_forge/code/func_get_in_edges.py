import math
import heapq
from collections import OrderedDict, defaultdict
import rustworkx as rx
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.controlflow import condition_resources
from qiskit.circuit.quantumregister import QuantumRegister, Qubit
from qiskit.circuit.classicalregister import ClassicalRegister, Clbit
from qiskit.dagcircuit.exceptions import DAGDependencyError
from qiskit.dagcircuit.dagdepnode import DAGDepNode
def get_in_edges(self, node_id):
    """
        Enumeration of all incoming edges for a given node.

        Args:
            node_id (int): label of considered node.

        Returns:
            List: corresponding incoming edges data.
        """
    return self._multi_graph.in_edges(node_id)