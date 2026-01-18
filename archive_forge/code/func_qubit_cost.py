import numpy as np
from scipy import integrate
from pennylane.operation import AnyWires, Operation
@staticmethod
def qubit_cost(n, eta, omega, error, br=7, charge=0, cubic=True, vectors=None):
    """Return the number of logical qubits needed to implement the first quantization
        algorithm.

        The expression for computing the cost is taken from Eq. (101) of
        [`arXiv:2204.11890v1 <https://arxiv.org/abs/2204.11890v1>`_].

        Args:
            n (int): number of plane waves
            eta (int): number of electrons
            omega (float): unit cell volume
            error (float): target error in the algorithm
            br (int): number of bits for ancilla qubit rotation
            charge (int): total electric charge of the system
            cubic (bool): True if the unit cell is cubic
            vectors (array[float]): lattice vectors

        Returns:
            int: number of logical qubits needed to implement the first quantization algorithm

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 169.69608
        >>> error = 0.01
        >>> qubit_cost(n, eta, omega, error)
        4377
        """
    if n <= 0:
        raise ValueError('The number of plane waves must be a positive number.')
    if eta <= 0 or not isinstance(eta, (int, np.integer)):
        raise ValueError('The number of electrons must be a positive integer.')
    if omega <= 0:
        raise ValueError('The unit cell volume must be a positive number.')
    if error <= 0.0:
        raise ValueError('The target error must be greater than zero.')
    if not isinstance(charge, int):
        raise ValueError('system charge must be an integer.')
    if not cubic:
        return FirstQuantization._qubit_cost_noncubic(n, eta, error, br, charge, vectors)
    lamb = FirstQuantization.norm(n, eta, omega, error, br=br, charge=charge)
    alpha = 0.01
    l_z = eta + charge
    l_nu = 2 * np.pi * n ** (2 / 3)
    n_p = np.ceil(np.log2(n ** (1 / 3) + 1))
    error_t = alpha * error
    error_r = alpha * error
    error_m = alpha * error
    n_t = int(np.log2(np.pi * lamb / error_t))
    n_r = int(np.log2(eta * l_z * l_nu / (error_r * omega ** (1 / 3))))
    n_m = int(np.log2(2 * eta / (error_m * np.pi * omega ** (1 / 3)) * (eta - 1 + 2 * l_z) * (7 * 2 ** (n_p + 1) - 9 * n_p - 11 - 3 * 2 ** (-1 * n_p))))
    alpha = 0.01
    error_qpe = np.sqrt(error ** 2 * (1 - (3 * alpha) ** 2))
    qubits = 3 * eta * n_p + 4 * n_m * n_p + 12 * n_p
    qubits += 2 * np.ceil(np.log2(np.ceil(np.pi * lamb / (2 * error_qpe)))) + 5 * n_m
    qubits += 2 * np.ceil(np.log2(eta)) + 3 * n_p ** 2 + np.ceil(np.log2(eta + 2 * l_z))
    qubits += np.maximum(5 * n_p + 1, 5 * n_r - 4) + np.maximum(n_t, n_r + 1) + 33
    return int(np.ceil(qubits))