import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def moment_integral(basis_a, basis_b, order, idx, normalize=True):
    """Return a function that computes the multipole moment integral for two contracted Gaussians.

    The multipole moment integral for two primitive Gaussian functions is computed as

    .. math::

        S^e = \\left \\langle G_i | q^e | G_j \\right \\rangle
                   \\left \\langle G_k | G_l \\right \\rangle
                   \\left \\langle G_m | G_n \\right \\rangle,

    where :math:`G_{i-n}` is a one-dimensional Gaussian function, :math:`q = x, y, z` is the
    coordinate at which the integral is evaluated and :math:`e` is a positive integer that is
    represented by the ``order`` argument. For contracted Gaussians, these integrals will be
    computed over primitive Gaussians, multiplied by the normalized contraction coefficients and
    finally summed over.

    The ``idx`` argument determines the coordinate :math:`q` at which the integral is computed. It
    can be :math:`0, 1, 2` for :math:`x, y, z` components, respectively.

    Args:
        basis_a (~qchem.basis_set.BasisFunction): left basis function
        basis_b (~qchem.basis_set.BasisFunction): right basis function
        order (integer): exponent of the position component
        idx (integer): index determining the dimension of the multipole moment integral
        normalize (bool): if True, the basis functions get normalized

    Returns:
        function: function that computes the multipole moment integral

    **Example**

    >>> symbols  = ['H', 'Li']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], requires_grad = True)
    >>> mol = qml.qchem.Molecule(symbols, geometry)
    >>> args = [mol.r] # initial values of the differentiable parameters
    >>> order, idx =  1, 0
    >>> moment_integral(mol.basis_set[0], mol.basis_set[1], order, idx)(*args)
    3.12846324e-01
    """

    def _moment_integral(*args):
        """Normalize and compute the multipole moment integral for two contracted Gaussians.

        Args:
            *args (array[float]): initial values of the differentiable parameters

        Returns:
            array[float]: the multipole moment integral between two contracted Gaussian orbitals
        """
        args_a = [arg[0] for arg in args]
        args_b = [arg[1] for arg in args]
        la = basis_a.l
        lb = basis_b.l
        alpha, ca, ra = _generate_params(basis_a.params, args_a)
        beta, cb, rb = _generate_params(basis_b.params, args_b)
        if basis_a.params[1].requires_grad or normalize:
            ca = ca * primitive_norm(basis_a.l, alpha)
            cb = cb * primitive_norm(basis_b.l, beta)
            na = contracted_norm(basis_a.l, alpha, ca)
            nb = contracted_norm(basis_b.l, beta, cb)
        else:
            na = nb = 1.0
        p = alpha[:, np.newaxis] + beta
        q = qml.math.sqrt(np.pi / p)
        r = (alpha[:, np.newaxis] * ra[:, np.newaxis, np.newaxis] + beta * rb[:, np.newaxis, np.newaxis]) / p
        i, j, k = qml.math.roll(qml.math.array([0, 2, 1]), idx)
        s = gaussian_moment(la[i], lb[i], ra[i], rb[i], alpha[:, np.newaxis], beta, order, r[i]) * expansion(la[j], lb[j], ra[j], rb[j], alpha[:, np.newaxis], beta, 0) * q * expansion(la[k], lb[k], ra[k], rb[k], alpha[:, np.newaxis], beta, 0) * q
        return (na * nb * (ca[:, np.newaxis] * cb) * s).sum()
    return _moment_integral