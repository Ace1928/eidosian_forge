import itertools
from typing import Iterable, Tuple, Dict
import networkx as nx
import cirq
def nx_qubit_layout(graph: nx.Graph) -> Dict[cirq.Qid, Tuple[float, float]]:
    """Return a layout for a graph for nodes which are qubits.

    This can be used in place of nx.spring_layout or other networkx layouts.
    GridQubits are positioned according to their row/col. LineQubits are
    positioned in a line.

    >>> import cirq.contrib.routing as ccr
    >>> import networkx as nx
    >>> import matplotlib.pyplot as plt
    >>> # Clear plot state to prevent issues with pyplot dimensionality.
    >>> plt.clf()
    >>> g = ccr.gridqubits_to_graph_device(cirq.GridQubit.rect(4,5))
    >>> pos = ccr.nx_qubit_layout(g)
    >>> nx.draw_networkx(g, pos=pos)

    """
    pos: Dict[cirq.Qid, Tuple[float, float]] = {}
    _node_to_i_cache = None
    for node in graph.nodes:
        if isinstance(node, cirq.GridQubit):
            pos[node] = (node.col, -node.row)
        elif isinstance(node, cirq.LineQubit):
            pos[node] = (node.x, 0.5)
        else:
            if _node_to_i_cache is None:
                _node_to_i_cache = {n: i for i, n in enumerate(sorted(graph.nodes))}
            pos[node] = (0.5, _node_to_i_cache[node] + 1)
    return pos