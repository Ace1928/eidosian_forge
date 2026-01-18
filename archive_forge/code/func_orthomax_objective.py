import numpy as np
def orthomax_objective(L=None, A=None, T=None, gamma=0, return_gradient=True):
    """
    Objective function for the orthomax family for orthogonal
    rotation wich minimizes the following objective:

    .. math::
        \\phi(L) = -\\frac{1}{4}(L\\circ L,(I-\\gamma C)(L\\circ L)),

    where :math:`0\\leq\\gamma\\leq1`, :math:`L` is a :math:`p\\times k` matrix,
    :math:`C` is a  :math:`p\\times p` matrix with elements equal to
    :math:`1/p`,
    :math:`(X,Y)=\\operatorname{Tr}(X^*Y)` is the Frobenius norm and
    :math:`\\circ` is the element-wise product or Hadamard product.

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix.

    The orthomax family is parametrized by the parameter :math:`\\gamma`:

    * :math:`\\gamma=0` corresponds to quartimax,
    * :math:`\\gamma=\\frac{1}{2}` corresponds to biquartimax,
    * :math:`\\gamma=1` corresponds to varimax,
    * :math:`\\gamma=\\frac{1}{p}` corresponds to equamax.

    Parameters
    ----------
    L : numpy matrix (default None)
        rotated factors, i.e., :math:`L=A(T^*)^{-1}=AT`
    A : numpy matrix (default None)
        non rotated factors
    T : numpy matrix (default None)
        rotation matrix
    gamma : float (default 0)
        a parameter
    return_gradient : bool (default True)
        toggles return of gradient
    """
    assert 0 <= gamma <= 1, 'Gamma should be between 0 and 1'
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method='orthogonal')
    p, k = L.shape
    L2 = L ** 2
    if np.isclose(gamma, 0):
        X = L2
    else:
        C = np.ones((p, p)) / p
        X = (np.eye(p) - gamma * C).dot(L2)
    phi = -np.sum(L2 * X) / 4
    if return_gradient:
        Gphi = -L * X
        return (phi, Gphi)
    else:
        return phi