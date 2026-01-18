import itertools as it
import numpy as np
import pennylane as qml
from .integrals import (
def mol_density_matrix(n_electron, c):
    """Compute the molecular density matrix.

    The density matrix :math:`P` is computed from the molecular orbital coefficients :math:`C` as

    .. math::

        P_{\\mu \\nu} = \\sum_{i=1}^{N} C_{\\mu i} C_{\\nu i},

    where :math:`N = N_{electrons} / 2` is the number of occupied orbitals. Note that the total
    density matrix is the sum of the :math:`\\alpha` and :math:`\\beta` density
    matrices, :math:`P = P^{\\alpha} + P^{\\beta}`.

    Args:
        n_electron (integer): number of electrons
        c (array[array[float]]): molecular orbital coefficients

    Returns:
        array[array[float]]: density matrix

    **Example**

    >>> c = np.array([[-0.54828771,  1.21848441], [-0.54828771, -1.21848441]])
    >>> n_electron = 2
    >>> mol_density_matrix(n_electron, c)
    array([[0.30061941, 0.30061941], [0.30061941, 0.30061941]])
    """
    p = qml.math.dot(c[:, :n_electron // 2], qml.math.conjugate(c[:, :n_electron // 2]).T)
    return p