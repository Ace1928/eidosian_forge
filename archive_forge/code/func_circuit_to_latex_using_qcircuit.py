from typing import TYPE_CHECKING
from cirq import circuits, ops
from cirq.contrib.qcircuit.qcircuit_diagram_info import (
def circuit_to_latex_using_qcircuit(circuit: 'cirq.Circuit', qubit_order: 'cirq.QubitOrderOrList'=ops.QubitOrder.DEFAULT) -> str:
    """Returns a QCircuit-based latex diagram of the given circuit.

    Args:
        circuit: The circuit to represent in latex.
        qubit_order: Determines the order of qubit wires in the diagram.

    Returns:
        Latex code for the diagram.
    """
    diagram = circuit.to_text_diagram_drawer(qubit_namer=qcircuit_qubit_namer, qubit_order=qubit_order, get_circuit_diagram_info=get_qcircuit_diagram_info, draw_moment_groups=False)
    return _render(diagram)