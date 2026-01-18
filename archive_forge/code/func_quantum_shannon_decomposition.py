from typing import List, Callable, TYPE_CHECKING
from scipy.linalg import cossin
import numpy as np
from cirq import ops
from cirq.linalg import decompositions, predicates
def quantum_shannon_decomposition(qubits: 'List[cirq.Qid]', u: np.ndarray) -> 'op_tree.OpTree':
    """Decomposes n-qubit unitary into CX/YPow/ZPow/CNOT gates, preserving global phase.

    The algorithm is described in Shende et al.:
    Synthesis of Quantum Logic Circuits. Tech. rep. 2006,
    https://arxiv.org/abs/quant-ph/0406176

    Args:
        qubits: List of qubits in order of significance
        u: Numpy array for unitary matrix representing gate to be decomposed

    Calls:
        (Base Case)
        1. _single_qubit_decomposition
            OR
        (Recursive Case)
        1. _msb_demuxer
        2. _multiplexed_cossin
        3. _msb_demuxer

    Yields:
        A single 2-qubit or 1-qubit operations from OP TREE
        composed from the set
           { CNOT, rz, ry, ZPowGate }

    Raises:
        ValueError: If the u matrix is non-unitary
        ValueError: If the u matrix is not of shape (2^n,2^n)
    """
    if not predicates.is_unitary(u):
        raise ValueError('Expected input matrix u to be unitary,                 but it fails cirq.is_unitary check')
    n = u.shape[0]
    if n & n - 1:
        raise ValueError(f'Expected input matrix u to be a (2^n x 2^n) shaped numpy array,                 but instead got shape {u.shape}')
    if n == 2:
        yield from _single_qubit_decomposition(qubits[0], u)
        return
    (u1, u2), theta, (v1, v2) = cossin(u, n / 2, n / 2, separate=True)
    yield from _msb_demuxer(qubits, v1, v2)
    yield from _multiplexed_cossin(qubits, theta, ops.ry)
    yield from _msb_demuxer(qubits, u1, u2)