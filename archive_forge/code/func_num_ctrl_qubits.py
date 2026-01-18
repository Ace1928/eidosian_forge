from __future__ import annotations
import copy
from typing import Optional, Union
from qiskit.circuit.exceptions import CircuitError
from .quantumcircuit import QuantumCircuit
from .gate import Gate
from .quantumregister import QuantumRegister
from ._utils import _ctrl_state_to_int
@num_ctrl_qubits.setter
def num_ctrl_qubits(self, num_ctrl_qubits):
    """Set the number of control qubits.

        Args:
            num_ctrl_qubits (int): The number of control qubits.

        Raises:
            CircuitError: ``num_ctrl_qubits`` is not an integer in ``[1, num_qubits]``.
        """
    if num_ctrl_qubits != int(num_ctrl_qubits):
        raise CircuitError('The number of control qubits must be an integer.')
    num_ctrl_qubits = int(num_ctrl_qubits)
    upper_limit = self.num_qubits - getattr(self.base_gate, 'num_qubits', 0)
    if num_ctrl_qubits < 1 or num_ctrl_qubits > upper_limit:
        limit = 'num_qubits' if self.base_gate is None else 'num_qubits - base_gate.num_qubits'
        raise CircuitError(f'The number of control qubits must be in `[1, {limit}]`.')
    self._num_ctrl_qubits = num_ctrl_qubits