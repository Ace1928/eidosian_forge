from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit._accelerate.quantum_circuit import CircuitData
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.parametertable import ParameterTable, ParameterView
def num_connected_components(self, unitary_only=False):
    if not self._is_built:
        self._build()
    return super().num_connected_components(unitary_only=unitary_only)