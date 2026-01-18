import numpy as np
from scipy import integrate
from pennylane.operation import AnyWires, Operation
@staticmethod
def success_prob(n, br):
    """Return the probability of success for state preparation.

        The expression for computing the probability of success is taken from Eqs. (59, 60) of
        [`PRX Quantum 2, 040332 (2021) <https://link.aps.org/doi/10.1103/PRXQuantum.2.040332>`_].

        Args:
            n (int): number of basis states to create an equal superposition for state preparation
            br (int): number of bits for ancilla qubit rotation

        Returns:
            float: probability of success for state preparation

        **Example**

        >>> n = 3
        >>> br = 8
        >>> success_prob(n, br)
        0.9999928850303523
        """
    if n <= 0:
        raise ValueError('The number of plane waves must be a positive number.')
    if br <= 0 or not isinstance(br, int):
        raise ValueError('br must be a positive integer.')
    c = n / 2 ** np.ceil(np.log2(n))
    d = 2 * np.pi / 2 ** br
    theta = d * np.round(1 / d * np.arcsin(np.sqrt(1 / (4 * c))))
    p = c * ((1 + (2 - 4 * c) * np.sin(theta) ** 2) ** 2 + np.sin(2 * theta) ** 2)
    return p