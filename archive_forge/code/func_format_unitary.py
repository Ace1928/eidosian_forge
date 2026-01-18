import numpy as np
from qiskit.exceptions import QiskitError
def format_unitary(mat, decimals=None):
    """Format unitary coming from the backend to present to the Qiskit user.

    Args:
        mat (list[list]): a list of list of [re, im] complex numbers
        decimals (int): the number of decimals in the statevector.
            If None, no rounding is done.

    Returns:
        list[list[complex]]: a matrix of complex numbers
    """
    from qiskit.quantum_info.operators.operator import Operator
    if isinstance(mat, Operator):
        if decimals:
            return Operator(np.around(mat.data, decimals=decimals), input_dims=mat.input_dims(), output_dims=mat.output_dims())
        return mat
    if isinstance(mat, np.ndarray):
        if decimals:
            return np.around(mat, decimals=decimals)
        return mat
    num_basis = len(mat)
    mat_complex = np.zeros((num_basis, num_basis), dtype=complex)
    for i, vec in enumerate(mat):
        mat_complex[i] = format_statevector(vec, decimals)
    return mat_complex