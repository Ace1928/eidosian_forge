import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def kinetic_integral(basis_a, basis_b, normalize=True):
    """Return a function that computes the kinetic integral for two contracted Gaussian functions.

    Args:
        basis_a (~qchem.basis_set.BasisFunction): first basis function
        basis_b (~qchem.basis_set.BasisFunction): second basis function
        normalize (bool): if True, the basis functions get normalized

    Returns:
        function: function that computes the kinetic integral

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404]], requires_grad = True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [mol.alpha]
    >>> kinetic_integral(mol.basis_set[0], mol.basis_set[1])(*args)
    0.38325367405312843
    """

    def _kinetic_integral(*args):
        """Compute the kinetic integral for two contracted Gaussian functions.

        Args:
            *args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the kinetic integral between two contracted Gaussian orbitals
        """
        args_a = [arg[0] for arg in args]
        args_b = [arg[1] for arg in args]
        alpha, ca, ra = _generate_params(basis_a.params, args_a)
        beta, cb, rb = _generate_params(basis_b.params, args_b)
        if basis_a.params[1].requires_grad or normalize:
            ca = ca * primitive_norm(basis_a.l, alpha)
            cb = cb * primitive_norm(basis_b.l, beta)
            na = contracted_norm(basis_a.l, alpha, ca)
            nb = contracted_norm(basis_b.l, beta, cb)
        else:
            na = nb = 1.0
        return na * nb * (ca[:, np.newaxis] * cb * gaussian_kinetic(basis_a.l, basis_b.l, ra, rb, alpha[:, np.newaxis], beta)).sum()
    return _kinetic_integral