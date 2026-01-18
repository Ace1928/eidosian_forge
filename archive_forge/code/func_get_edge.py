import heapq
from collections import OrderedDict
from functools import partial
from typing import Sequence, Callable
import networkx as nx
from networkx.drawing.nx_pydot import to_pydot
import pennylane as qml
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
from pennylane.transforms import transform
def get_edge(self, node_in, node_out):
    """Get the edge between two nodes if it exists.

        Args:
            node_in (int): Id of the ingoing node.
            node_out (int): Id of the outgoing node.

        Returns:
            dict or None: Default weight is 0, it returns None when there is no edge.
        """
    return self._multi_graph.get_edge_data(node_in, node_out)