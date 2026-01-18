import itertools as it
import numpy as np
import pennylane as qml
from .integrals import (
def kinetic_matrix(basis_functions):
    """Return a function that computes the kinetic matrix for a given set of basis functions.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions

    Returns:
        function: function that computes the kinetic matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> kinetic_matrix(mol.basis_set)(*args)
    array([[0.76003189, 0.38325367], [0.38325367, 0.76003189]])
    """

    def kinetic(*args):
        """Construct the kinetic matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the kinetic matrix
        """
        n = len(basis_functions)
        matrix = qml.math.zeros((n, n))
        for (i, a), (j, b) in it.combinations_with_replacement(enumerate(basis_functions), r=2):
            args_ab = []
            if args:
                args_ab.extend(([arg[i], arg[j]] for arg in args))
            integral = kinetic_integral(a, b, normalize=False)(*args_ab)
            o = qml.math.zeros((n, n))
            o[i, j] = o[j, i] = 1.0
            matrix = matrix + integral * o
        return matrix
    return kinetic