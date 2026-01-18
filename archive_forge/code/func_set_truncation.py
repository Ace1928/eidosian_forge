from __future__ import annotations
import re
from typing import Literal, TYPE_CHECKING
import numpy as np
from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.delay import Delay
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli, _count_y
@classmethod
def set_truncation(cls, val: int):
    """Set the max number of Pauli characters to display before truncation/

        Args:
            val (int): the number of characters.

        .. note::

            Truncation will be disabled if the truncation value is set to 0.
        """
    cls.__truncate__ = int(val)