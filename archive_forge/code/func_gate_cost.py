import numpy as np
from scipy import integrate
from pennylane.operation import AnyWires, Operation
@staticmethod
def gate_cost(n, eta, omega, error, br=7, charge=0, cubic=True, vectors=None):
    """Return the total number of Toffoli gates needed to implement the first quantization
        algorithm.

        The expression for computing the cost is taken from Eq. (125) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

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
            int: the number of Toffoli gates needed to implement the first quantization algorithm

        **Example**

        >>> n = 100000
        >>> eta = 156
        >>> omega = 169.69608
        >>> error = 0.01
        >>> gate_cost(n, eta, omega, error)
        3676557345574
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
    if br <= 0 or not isinstance(br, int):
        raise ValueError('br must be a positive integer.')
    e_cost = FirstQuantization.estimation_cost(n, eta, omega, error, br=br, charge=charge, cubic=cubic, vectors=vectors)
    if cubic:
        u_cost = FirstQuantization.unitary_cost(n, eta, omega, error, br, charge)
    else:
        u_cost = FirstQuantization._unitary_cost_noncubic(n, eta, error, br, charge, vectors)
    return e_cost * u_cost