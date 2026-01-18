from typing import cast
from cirq import circuits, ops, protocols
from cirq.contrib import circuitdag
def pauli_string_dag_from_circuit(circuit: circuits.Circuit) -> circuitdag.CircuitDag:
    return circuitdag.CircuitDag.from_circuit(circuit, pauli_string_reorder_pred)