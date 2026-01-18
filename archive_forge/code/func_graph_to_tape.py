import copy
from itertools import product
from typing import Callable, List, Sequence, Tuple, Union
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import expval
from pennylane.measurements import ExpectationMP, MeasurementProcess, SampleMP
from pennylane.operation import Operator, Tensor
from pennylane.ops.meta import WireCut
from pennylane.pauli import string_to_pauli_word
from pennylane.queuing import WrappedObj
from pennylane.tape import QuantumScript, QuantumTape
from pennylane.wires import Wires
from .utils import MeasureNode, PrepareNode
def graph_to_tape(graph: MultiDiGraph) -> QuantumTape:
    """
    Converts a directed multigraph to the corresponding :class:`~.QuantumTape`.

    To account for the possibility of needing to perform mid-circuit measurements, if any operations
    follow a :class:`MeasureNode` operation on a given wire then these operations are mapped to a
    new wire.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        graph (nx.MultiDiGraph): directed multigraph to be converted to a tape

    Returns:
        QuantumTape: the quantum tape corresponding to the input graph

    **Example**

    Consider the following circuit:

    .. code-block:: python

        ops = [
            qml.RX(0.4, wires=0),
            qml.RY(0.5, wires=1),
            qml.CNOT(wires=[0, 1]),
            qml.qcut.MeasureNode(wires=1),
            qml.qcut.PrepareNode(wires=1),
            qml.CNOT(wires=[1, 0]),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    This circuit contains operations that follow a :class:`~.MeasureNode`. These operations will
    subsequently act on wire ``2`` instead of wire ``1``:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> tape = qml.qcut.graph_to_tape(graph)
    >>> print(tape.draw())
    0: ──RX──────────╭●──────────────╭X─┤  <Z>
    1: ──RY──────────╰X──MeasureNode─│──┤
    2: ──PrepareNode─────────────────╰●─┤

    """
    wires = Wires.all_wires([n.obj.wires for n in graph.nodes])
    ordered_ops = sorted([(order, op.obj) for op, order in graph.nodes(data='order')], key=lambda x: x[0])
    wire_map = {w: w for w in wires}
    reverse_wire_map = {v: k for k, v in wire_map.items()}
    copy_ops = [copy.copy(op) for _, op in ordered_ops if not isinstance(op, MeasurementProcess)]
    copy_meas = [copy.copy(op) for _, op in ordered_ops if isinstance(op, MeasurementProcess)]
    observables = []
    operations_from_graph = []
    measurements_from_graph = []
    for op in copy_ops:
        op = qml.map_wires(op, wire_map=wire_map, queue=False)
        operations_from_graph.append(op)
        if isinstance(op, MeasureNode):
            assert len(op.wires) == 1
            measured_wire = op.wires[0]
            new_wire = _find_new_wire(wires)
            wires += new_wire
            original_wire = reverse_wire_map[measured_wire]
            wire_map[original_wire] = new_wire
            reverse_wire_map[new_wire] = original_wire
    if copy_meas:
        measurement_types = {type(meas) for meas in copy_meas}
        if len(measurement_types) > 1:
            raise ValueError('Only a single return type can be used for measurement nodes in graph_to_tape')
        measurement_type = measurement_types.pop()
        if measurement_type not in {SampleMP, ExpectationMP}:
            raise ValueError('Invalid return type. Only expectation value and sampling measurements are supported in graph_to_tape')
        for meas in copy_meas:
            meas = qml.map_wires(meas, wire_map=wire_map)
            obs = meas.obs
            observables.append(obs)
            if measurement_type is SampleMP:
                measurements_from_graph.append(meas)
        if measurement_type is ExpectationMP:
            if len(observables) > 1:
                measurements_from_graph.append(qml.expval(Tensor(*observables)))
            else:
                measurements_from_graph.append(qml.expval(obs))
    return QuantumScript(ops=operations_from_graph, measurements=measurements_from_graph)