import numpy as np
def oblimin_objective(L=None, A=None, T=None, gamma=0, rotation_method='orthogonal', return_gradient=True):
    """
    Objective function for the oblimin family for orthogonal or
    oblique rotation wich minimizes:

    .. math::
        \\phi(L) = \\frac{1}{4}(L\\circ L,(I-\\gamma C)(L\\circ L)N),

    where :math:`L` is a :math:`p\\times k` matrix, :math:`N` is
    :math:`k\\times k`
    matrix with zeros on the diagonal and ones elsewhere, :math:`C` is a
    :math:`p\\times p` matrix with elements equal to :math:`1/p`,
    :math:`(X,Y)=\\operatorname{Tr}(X^*Y)` is the Frobenius norm and
    :math:`\\circ`
    is the element-wise product or Hadamard product.

    The gradient is given by

    .. math::
        L\\circ\\left[(I-\\gamma C) (L \\circ L)N\\right].

    Either :math:`L` should be provided or :math:`A` and :math:`T` should be
    provided.

    For orthogonal rotations :math:`L` satisfies

    .. math::
        L =  AT,

    where :math:`T` is an orthogonal matrix. For oblique rotations :math:`L`
    satisfies

    .. math::
        L =  A(T^*)^{-1},

    where :math:`T` is a normal matrix.

    The oblimin family is parametrized by the parameter :math:`\\gamma`. For
    orthogonal rotations:

    * :math:`\\gamma=0` corresponds to quartimax,
    * :math:`\\gamma=\\frac{1}{2}` corresponds to biquartimax,
    * :math:`\\gamma=1` corresponds to varimax,
    * :math:`\\gamma=\\frac{1}{p}` corresponds to equamax.
    For oblique rotations rotations:

    * :math:`\\gamma=0` corresponds to quartimin,
    * :math:`\\gamma=\\frac{1}{2}` corresponds to biquartimin.

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
    rotation_method : str
        should be one of {orthogonal, oblique}
    return_gradient : bool (default True)
        toggles return of gradient
    """
    if L is None:
        assert A is not None and T is not None
        L = rotateA(A, T, rotation_method=rotation_method)
    p, k = L.shape
    L2 = L ** 2
    N = np.ones((k, k)) - np.eye(k)
    if np.isclose(gamma, 0):
        X = L2.dot(N)
    else:
        C = np.ones((p, p)) / p
        X = (np.eye(p) - gamma * C).dot(L2).dot(N)
    phi = np.sum(L2 * X) / 4
    if return_gradient:
        Gphi = L * X
        return (phi, Gphi)
    else:
        return phi