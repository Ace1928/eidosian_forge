from __future__ import annotations
import numpy as np
from qiskit._accelerate import euler_one_qubit_decomposer
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import Qubit
from qiskit.circuit.library.standard_gates import (
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.circuit.gate import Gate
from qiskit.quantum_info.operators.operator import Operator
def build_circuit(self, gates, global_phase):
    """Return the circuit or dag object from a list of gates."""
    qr = [Qubit()]
    lookup_gate = False
    if len(gates) > 0 and isinstance(gates[0], tuple):
        lookup_gate = True
    if self.use_dag:
        from qiskit.dagcircuit import dagcircuit
        dag = dagcircuit.DAGCircuit()
        dag.global_phase = global_phase
        dag.add_qubits(qr)
        for gate_entry in gates:
            if lookup_gate:
                gate = NAME_MAP[gate_entry[0]](*gate_entry[1])
            else:
                gate = gate_entry
            dag.apply_operation_back(gate, (qr[0],), check=False)
        return dag
    else:
        circuit = QuantumCircuit(qr, global_phase=global_phase)
        for gate_entry in gates:
            if lookup_gate:
                gate = NAME_MAP[gate_entry[0]](*gate_entry[1])
            else:
                gate = gate_entry
            circuit._append(gate, [qr[0]], [])
        return circuit