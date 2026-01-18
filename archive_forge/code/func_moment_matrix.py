import itertools as it
import numpy as np
import pennylane as qml
from .integrals import (
def moment_matrix(basis_functions, order, idx):
    """Return a function that computes the multipole moment matrix for a set of basis functions.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions
        order (integer): exponent of the position component
        idx (integer): index determining the dimension of the multipole moment integral

    Returns:
        function: function that computes the multipole moment matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> order, idx = 1, 0
    >>> moment_matrix(mol.basis_set, order, idx)(*args)
    tensor([[0.0, 0.4627777], [0.4627777, 2.0]], requires_grad=True)
    """

    def _moment_matrix(*args):
        """Construct the multipole moment matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the multipole moment matrix
        """
        n = len(basis_functions)
        matrix = qml.math.zeros((n, n))
        for (i, a), (j, b) in it.combinations_with_replacement(enumerate(basis_functions), r=2):
            args_ab = []
            if args:
                args_ab.extend(([arg[i], arg[j]] for arg in args))
            integral = moment_integral(a, b, order, idx, normalize=False)(*args_ab)
            o = qml.math.zeros((n, n))
            o[i, j] = o[j, i] = 1.0
            matrix = matrix + integral * o
        return matrix
    return _moment_matrix