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
def is_scalar_nonsparse_matrix(circuit, nqubits, identity_only, eps=None):
    """Checks if a given circuit, in matrix form, is equivalent to
    a scalar value.

    Parameters
    ==========

    circuit : Gate tuple
        Sequence of quantum gates representing a quantum circuit
    nqubits : int
        Number of qubits in the circuit
    identity_only : bool
        Check for only identity matrices
    eps : number
        This argument is ignored. It is just for signature compatibility with
        is_scalar_sparse_matrix.

    Note: Used in situations when is_scalar_sparse_matrix has bugs
    """
    matrix = represent(Mul(*circuit), nqubits=nqubits)
    if isinstance(matrix, Number):
        return matrix == 1 if identity_only else True
    else:
        matrix_trace = matrix.trace()
        adjusted_matrix_trace = matrix_trace / matrix[0] if not identity_only else matrix_trace
        is_identity = equal_valued(matrix[0], 1) if identity_only else True
        has_correct_trace = adjusted_matrix_trace == pow(2, nqubits)
        return bool(matrix.is_diagonal() and has_correct_trace and is_identity)