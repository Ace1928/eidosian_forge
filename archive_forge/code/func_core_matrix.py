import itertools as it
import numpy as np
import pennylane as qml
from .integrals import (
def core_matrix(basis_functions, charges, r):
    """Return a function that computes the core matrix for a given set of basis functions.

    The core matrix is computed as a sum of the kinetic and electron-nuclear attraction matrices.

    Args:
        basis_functions (list[~qchem.basis_set.BasisFunction]): basis functions
        charges (list[int]): nuclear charges
        r (array[float]): nuclear positions

    Returns:
        function: function that computes the core matrix

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> core_matrix(mol.basis_set, mol.nuclear_charges, mol.coordinates)(*args)
    array([[-1.27848869, -1.21916299], [-1.21916299, -1.27848869]])
    """

    def core(*args):
        """Construct the core matrix for a given set of basis functions.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            array[array[float]]: the core matrix
        """
        if r.requires_grad:
            t = kinetic_matrix(basis_functions)(*args[1:])
        else:
            t = kinetic_matrix(basis_functions)(*args)
        a = attraction_matrix(basis_functions, charges, r)(*args)
        return t + a
    return core