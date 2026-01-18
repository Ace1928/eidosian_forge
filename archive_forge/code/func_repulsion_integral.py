import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def repulsion_integral(basis_a, basis_b, basis_c, basis_d, normalize=True):
    """Return a function that computes the electron-electron repulsion integral for four contracted
    Gaussian functions.

    Args:
        basis_a (~qchem.basis_set.BasisFunction): first basis function
        basis_b (~qchem.basis_set.BasisFunction): second basis function
        basis_c (~qchem.basis_set.BasisFunction): third basis function
        basis_d (~qchem.basis_set.BasisFunction): fourth basis function
        normalize (bool): if True, the basis functions get normalized

    Returns:
        function: function that computes the electron repulsion integral

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404],
    >>>                   [3.425250914, 0.6239137298, 0.168855404]], requires_grad = True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> basis_a = mol.basis_set[0]
    >>> basis_b = mol.basis_set[1]
    >>> args = [mol.alpha]
    >>> repulsion_integral(basis_a, basis_b, basis_a, basis_b)(*args)
    0.45590152106593573
    """

    def _repulsion_integral(*args):
        """Compute the electron-electron repulsion integral for four contracted Gaussian functions.

        Args:
            *args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the electron repulsion integral between four contracted Gaussian functions
        """
        args_a = [arg[0] for arg in args]
        args_b = [arg[1] for arg in args]
        args_c = [arg[2] for arg in args]
        args_d = [arg[3] for arg in args]
        alpha, ca, ra = _generate_params(basis_a.params, args_a)
        beta, cb, rb = _generate_params(basis_b.params, args_b)
        gamma, cc, rc = _generate_params(basis_c.params, args_c)
        delta, cd, rd = _generate_params(basis_d.params, args_d)
        if basis_a.params[1].requires_grad or normalize:
            ca = ca * primitive_norm(basis_a.l, alpha)
            cb = cb * primitive_norm(basis_b.l, beta)
            cc = cc * primitive_norm(basis_c.l, gamma)
            cd = cd * primitive_norm(basis_d.l, delta)
            n1 = contracted_norm(basis_a.l, alpha, ca)
            n2 = contracted_norm(basis_b.l, beta, cb)
            n3 = contracted_norm(basis_c.l, gamma, cc)
            n4 = contracted_norm(basis_d.l, delta, cd)
        else:
            n1 = n2 = n3 = n4 = 1.0
        e = n1 * n2 * n3 * n4 * (ca * cb[:, np.newaxis] * cc[:, np.newaxis, np.newaxis] * cd[:, np.newaxis, np.newaxis, np.newaxis] * electron_repulsion(basis_a.l, basis_b.l, basis_c.l, basis_d.l, ra, rb, rc, rd, alpha, beta[:, np.newaxis], gamma[:, np.newaxis, np.newaxis], delta[:, np.newaxis, np.newaxis, np.newaxis])).sum()
        return e
    return _repulsion_integral