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
def get_all_edges(self):
    """
        Enumeration of all edges.

        Returns:
            List: corresponding to the label.
        """
    return [(src, dest, data) for src_node in self._multi_graph.nodes() for src, dest, data in self._multi_graph.out_edges(src_node.node_id)]