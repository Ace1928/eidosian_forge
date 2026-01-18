from __future__ import annotations
import functools
import itertools
import re
from typing import Literal
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.library.standard_gates import HGate, IGate, SGate, XGate, YGate, ZGate
from qiskit.circuit.operation import Operation
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.mixins import AdjointMixin, generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import _count_y
from .base_pauli import BasePauli
from .clifford_circuits import _append_circuit, _append_operation
@staticmethod
def from_label(label: str) -> Clifford:
    """Return a tensor product of single-qubit Clifford gates.

        Args:
            label (string): single-qubit operator string.

        Returns:
            Clifford: The N-qubit Clifford operator.

        Raises:
            QiskitError: if the label contains invalid characters.

        Additional Information:
            The labels correspond to the single-qubit Cliffords are

            * - Label
              - Stabilizer
              - Destabilizer
            * - ``"I"``
              - +Z
              - +X
            * - ``"X"``
              - -Z
              - +X
            * - ``"Y"``
              - -Z
              - -X
            * - ``"Z"``
              - +Z
              - -X
            * - ``"H"``
              - +X
              - +Z
            * - ``"S"``
              - +Z
              - +Y
        """
    label_gates = {'I': IGate(), 'X': XGate(), 'Y': YGate(), 'Z': ZGate(), 'H': HGate(), 'S': SGate()}
    if re.match('^[IXYZHS\\-+]+$', label) is None:
        raise QiskitError('Label contains invalid characters.')
    num_qubits = len(label)
    op = Clifford(np.eye(2 * num_qubits, dtype=bool))
    for qubit, char in enumerate(reversed(label)):
        op = _append_operation(op, label_gates[char], qargs=[qubit])
    return op