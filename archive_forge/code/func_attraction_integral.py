import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def attraction_integral(r, basis_a, basis_b, normalize=True):
    """Return a function that computes the nuclear attraction integral for two contracted Gaussian
    functions.

    Args:
        r (array[float]): position vector of nucleus
        basis_a (~qchem.basis_set.BasisFunction): first basis function
        basis_b (~qchem.basis_set.BasisFunction): second basis function
        normalize (bool): if True, the basis functions get normalized

    Returns:
        function: function that computes the electron-nuclear attraction integral

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404]], requires_grad = True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> basis_a = mol.basis_set[0]
    >>> basis_b = mol.basis_set[1]
    >>> args = [mol.alpha]
    >>> attraction_integral(geometry[0], basis_a, basis_b)(*args)
    0.801208332328965
    """

    def _attraction_integral(*args):
        """Compute the electron-nuclear attraction integral for two contracted Gaussian functions.

        Args:
            *args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the electron-nuclear attraction integral
        """
        if r.requires_grad:
            coor = args[0]
            args_a = [arg[0] for arg in args[1:]]
            args_b = [arg[1] for arg in args[1:]]
        else:
            coor = r
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
        v = na * nb * (ca * cb[:, np.newaxis] * nuclear_attraction(basis_a.l, basis_b.l, ra, rb, alpha, beta[:, np.newaxis], coor)).sum()
        return v
    return _attraction_integral