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
@classmethod
def from_permutation(cls, permutation_gate):
    """Create a Clifford from a PermutationGate.

        Args:
            permutation_gate (PermutationGate): A permutation to be converted.

        Returns:
            Clifford: the Clifford object for this permutation.
        """
    pat = permutation_gate.pattern
    dim = len(pat)
    symplectic_mat = np.zeros((2 * dim, 2 * dim), dtype=int)
    for i, j in enumerate(pat):
        symplectic_mat[j, i] = True
        symplectic_mat[j + dim, i + dim] = True
    phase = np.zeros(2 * dim, dtype=bool)
    tableau = cls._stack_table_phase(symplectic_mat, phase)
    return tableau