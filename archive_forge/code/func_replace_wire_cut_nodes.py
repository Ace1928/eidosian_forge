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
def replace_wire_cut_nodes(graph: MultiDiGraph):
    """
    Replace each :class:`~.WireCut` node in the graph with a
    :class:`~.MeasureNode` and :class:`~.PrepareNode`.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        graph (nx.MultiDiGraph): The graph containing the :class:`~.WireCut` nodes
            to be replaced

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

    We can find the circuit graph and remove all the wire cut nodes using:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> qml.qcut.replace_wire_cut_nodes(graph)
    """
    for op in list(graph.nodes):
        if isinstance(op.obj, WireCut):
            replace_wire_cut_node(op.obj, graph)