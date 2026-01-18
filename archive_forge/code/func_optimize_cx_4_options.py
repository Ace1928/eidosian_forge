import copy
from typing import Callable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.circuit.exceptions import CircuitError
from . import calc_inverse_matrix, check_invertible_binary_matrix
def optimize_cx_4_options(function: Callable, mat: np.ndarray, optimize_count: bool=True):
    """Get the best implementation of a circuit implementing a binary invertible matrix M,
    by considering all four options: M,M^(-1),M^T,M^(-1)^T.
    Optimizing either the CX count or the depth.

    Args:
        function: the synthesis function.
        mat: a binary invertible matrix.
        optimize_count: True if the number of CX gates in optimize, False if the depth is optimized.

    Returns:
        QuantumCircuit: an optimized :class:`.QuantumCircuit`, has the best depth or CX count of
            the four options.

    Raises:
        QiskitError: if mat is not an invertible matrix.
    """
    if not check_invertible_binary_matrix(mat):
        raise QiskitError('The matrix is not invertible.')
    qc = function(mat)
    best_qc = qc
    best_depth = qc.depth()
    best_count = qc.count_ops()['cx']
    for i in range(1, 4):
        mat_cpy = copy.deepcopy(mat)
        if i == 1:
            mat_cpy = calc_inverse_matrix(mat_cpy)
            qc = function(mat_cpy)
            qc = qc.inverse()
        elif i == 2:
            mat_cpy = np.transpose(mat_cpy)
            qc = function(mat_cpy)
            qc = transpose_cx_circ(qc)
        elif i == 3:
            mat_cpy = calc_inverse_matrix(np.transpose(mat_cpy))
            qc = function(mat_cpy)
            qc = transpose_cx_circ(qc)
            qc = qc.inverse()
        new_depth = qc.depth()
        new_count = qc.count_ops()['cx']
        better_count = optimize_count and best_count > new_count or (not optimize_count and best_depth == new_depth and (best_count > new_count))
        better_depth = not optimize_count and best_depth > new_depth or (optimize_count and best_count == new_count and (best_depth > new_depth))
        if better_count or better_depth:
            best_count = new_count
            best_depth = new_depth
            best_qc = qc
    return best_qc