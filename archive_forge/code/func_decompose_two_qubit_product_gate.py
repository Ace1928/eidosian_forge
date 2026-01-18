from __future__ import annotations
import cmath
import math
import io
import base64
import warnings
from typing import ClassVar, Optional, Type
import logging
import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.library.standard_gates import CXGate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.two_qubit.weyl import transform_to_magic_basis
from qiskit.synthesis.one_qubit.one_qubit_decompose import (
from qiskit._accelerate import two_qubit_decompose
def decompose_two_qubit_product_gate(special_unitary_matrix: np.ndarray):
    """Decompose :math:`U = U_l \\otimes U_r` where :math:`U \\in SU(4)`,
    and :math:`U_l,~U_r \\in SU(2)`.

    Args:
        special_unitary_matrix: special unitary matrix to decompose
    Raises:
        QiskitError: if decomposition isn't possible.
    """
    special_unitary_matrix = np.asarray(special_unitary_matrix, dtype=complex)
    R = special_unitary_matrix[:2, :2].copy()
    detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        R = special_unitary_matrix[2:, :2].copy()
        detR = R[0, 0] * R[1, 1] - R[0, 1] * R[1, 0]
    if abs(detR) < 0.1:
        raise QiskitError('decompose_two_qubit_product_gate: unable to decompose: detR < 0.1')
    R /= np.sqrt(detR)
    temp = np.kron(np.eye(2), R.T.conj())
    temp = special_unitary_matrix.dot(temp)
    L = temp[::2, ::2]
    detL = L[0, 0] * L[1, 1] - L[0, 1] * L[1, 0]
    if abs(detL) < 0.9:
        raise QiskitError('decompose_two_qubit_product_gate: unable to decompose: detL < 0.9')
    L /= np.sqrt(detL)
    phase = cmath.phase(detL) / 2
    temp = np.kron(L, R)
    deviation = abs(abs(temp.conj().T.dot(special_unitary_matrix).trace()) - 4)
    if deviation > 1e-13:
        raise QiskitError('decompose_two_qubit_product_gate: decomposition failed: deviation too large: {}'.format(deviation))
    return (L, R, phase)