from collections import deque
from sympy.core.random import randint
from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
def is_scalar_sparse_matrix(circuit, nqubits, identity_only, eps=1e-11):
    """Checks if a given scipy.sparse matrix is a scalar matrix.

    A scalar matrix is such that B = bI, where B is the scalar
    matrix, b is some scalar multiple, and I is the identity
    matrix.  A scalar matrix would have only the element b along
    it's main diagonal and zeroes elsewhere.

    Parameters
    ==========

    circuit : Gate tuple
        Sequence of quantum gates representing a quantum circuit
    nqubits : int
        Number of qubits in the circuit
    identity_only : bool
        Check for only identity matrices
    eps : number
        The tolerance value for zeroing out elements in the matrix.
        Values in the range [-eps, +eps] will be changed to a zero.
    """
    if not np or not scipy:
        pass
    matrix = represent(Mul(*circuit), nqubits=nqubits, format='scipy.sparse')
    if isinstance(matrix, int):
        return matrix == 1 if identity_only else True
    else:
        dense_matrix = matrix.todense().getA()
        bool_real = np.logical_and(dense_matrix.real > -eps, dense_matrix.real < eps)
        bool_imag = np.logical_and(dense_matrix.imag > -eps, dense_matrix.imag < eps)
        corrected_real = np.where(bool_real, 0.0, dense_matrix.real)
        corrected_imag = np.where(bool_imag, 0.0, dense_matrix.imag)
        corrected_imag = corrected_imag * complex(1j)
        corrected_dense = corrected_real + corrected_imag
        row_indices = corrected_dense.nonzero()[0]
        col_indices = corrected_dense.nonzero()[1]
        bool_indices = row_indices == col_indices
        is_diagonal = bool_indices.all()
        first_element = corrected_dense[0][0]
        if first_element == 0.0 + 0j:
            return False
        trace_of_corrected = (corrected_dense / first_element).trace()
        expected_trace = pow(2, nqubits)
        has_correct_trace = trace_of_corrected == expected_trace
        real_is_one = abs(first_element.real - 1.0) < eps
        imag_is_zero = abs(first_element.imag) < eps
        is_one = real_is_one and imag_is_zero
        is_identity = is_one if identity_only else True
        return bool(is_diagonal and has_correct_trace and is_identity)