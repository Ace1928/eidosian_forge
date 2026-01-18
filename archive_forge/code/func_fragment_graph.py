import uuid
from typing import Any, Callable, Sequence, Tuple
import warnings
import numpy as np
from networkx import MultiDiGraph, has_path, weakly_connected_components
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.meta import WireCut
from pennylane.queuing import WrappedObj
from pennylane.operation import Operation
from .kahypar import kahypar_cut
from .cutstrategy import CutStrategy
def fragment_graph(graph: MultiDiGraph) -> Tuple[Tuple[MultiDiGraph], MultiDiGraph]:
    """
    Fragments a graph into a collection of subgraphs as well as returning
    the communication (`quotient <https://en.wikipedia.org/wiki/Quotient_graph>`__)
    graph.

    The input ``graph`` is fragmented by disconnecting each :class:`~.MeasureNode` and
    :class:`~.PrepareNode` pair and finding the resultant disconnected subgraph fragments.
    Each node of the communication graph represents a subgraph fragment and the edges
    denote the flow of qubits between fragments due to the removed :class:`~.MeasureNode` and
    :class:`~.PrepareNode` pairs.

    .. note::

        This operation is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        graph (nx.MultiDiGraph): directed multigraph containing measure and prepare
            nodes at cut locations

    Returns:
        Tuple[Tuple[nx.MultiDiGraph], nx.MultiDiGraph]: the subgraphs of the cut graph
        and the communication graph.

    **Example**

    Consider the following circuit with manually-placed wire cuts:

    .. code-block:: python

        wire_cut_0 = qml.WireCut(wires=0)
        wire_cut_1 = qml.WireCut(wires=1)
        multi_wire_cut = qml.WireCut(wires=[0, 1])

        ops = [
            qml.RX(0.4, wires=0),
            wire_cut_0,
            qml.RY(0.5, wires=0),
            wire_cut_1,
            qml.CNOT(wires=[0, 1]),
            multi_wire_cut,
            qml.RZ(0.6, wires=1),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can find the corresponding graph, remove all the wire cut nodes, and
    find the subgraphs and communication graph by using:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> qml.qcut.replace_wire_cut_nodes(graph)
    >>> qml.qcut.fragment_graph(graph)
    ((<networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b2311940>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b2311c10>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e2820>,
      <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e27f0>),
     <networkx.classes.multidigraph.MultiDiGraph object at 0x7fb3b23e26a0>)
    """
    graph_copy = graph.copy()
    cut_edges = []
    measure_nodes = [n for n in graph.nodes if isinstance(n.obj, MeasurementProcess)]
    for node1, node2, wire_key in graph.edges:
        if isinstance(node1.obj, MeasureNode):
            assert isinstance(node2.obj, PrepareNode)
            cut_edges.append((node1, node2, wire_key))
            graph_copy.remove_edge(node1, node2, key=wire_key)
    subgraph_nodes = weakly_connected_components(graph_copy)
    subgraphs = tuple((MultiDiGraph(graph_copy.subgraph(n)) for n in subgraph_nodes))
    communication_graph = MultiDiGraph()
    communication_graph.add_nodes_from(range(len(subgraphs)))
    for node1, node2, _ in cut_edges:
        for i, subgraph in enumerate(subgraphs):
            if subgraph.has_node(node1):
                start_fragment = i
            if subgraph.has_node(node2):
                end_fragment = i
        if start_fragment != end_fragment:
            communication_graph.add_edge(start_fragment, end_fragment, pair=(node1, node2))
        else:
            subgraphs[start_fragment].remove_node(node1)
            subgraphs[end_fragment].remove_node(node2)
    terminal_indices = [i for i, s in enumerate(subgraphs) for n in measure_nodes if s.has_node(n)]
    subgraphs_connected_to_measurements = []
    subgraphs_indices_to_remove = []
    prepare_nodes_removed = []
    for i, s in enumerate(subgraphs):
        if any((has_path(communication_graph, i, t) for t in terminal_indices)):
            subgraphs_connected_to_measurements.append(s)
        else:
            subgraphs_indices_to_remove.append(i)
            prepare_nodes_removed.extend([n for n in s.nodes if isinstance(n.obj, PrepareNode)])
    measure_nodes_to_remove = [m for p in prepare_nodes_removed for m, p_, _ in cut_edges if p is p_]
    communication_graph.remove_nodes_from(subgraphs_indices_to_remove)
    for m in measure_nodes_to_remove:
        for s in subgraphs_connected_to_measurements:
            if s.has_node(m):
                s.remove_node(m)
    return (subgraphs_connected_to_measurements, communication_graph)