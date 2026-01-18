from __future__ import annotations
import numpy as np
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.quantum_info import SparsePauliOp
from .evolved_operator_ansatz import EvolvedOperatorAnsatz, _is_pauli_identity
@cost_operator.setter
def cost_operator(self, cost_operator) -> None:
    """Sets cost operator.

        Args:
            cost_operator (BaseOperator or OperatorBase, optional): cost operator to set.
        """
    self._cost_operator = cost_operator
    self.qregs = [QuantumRegister(self.num_qubits, name='q')]
    self._invalidate()