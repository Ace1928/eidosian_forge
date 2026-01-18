from scipy import stats, integrate, special
import numpy as np
def is_orthonormal_cont(polys, lower, upper, rtol=0, atol=1e-08):
    """check whether functions are orthonormal

    Parameters
    ----------
    polys : list of polynomials or function

    Returns
    -------
    is_orthonormal : bool
        is False if the innerproducts are not close to 0 or 1

    Notes
    -----
    this stops as soon as the first deviation from orthonormality is found.

    Examples
    --------
    >>> from scipy.special import chebyt
    >>> polys = [chebyt(i) for i in range(4)]
    >>> r, e = inner_cont(polys, -1, 1)
    >>> r
    array([[ 2.        ,  0.        , -0.66666667,  0.        ],
           [ 0.        ,  0.66666667,  0.        , -0.4       ],
           [-0.66666667,  0.        ,  0.93333333,  0.        ],
           [ 0.        , -0.4       ,  0.        ,  0.97142857]])
    >>> is_orthonormal_cont(polys, -1, 1, atol=1e-6)
    False

    >>> polys = [ChebyTPoly(i) for i in range(4)]
    >>> r, e = inner_cont(polys, -1, 1)
    >>> r
    array([[  1.00000000e+00,   0.00000000e+00,  -9.31270888e-14,
              0.00000000e+00],
           [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
             -9.47850712e-15],
           [ -9.31270888e-14,   0.00000000e+00,   1.00000000e+00,
              0.00000000e+00],
           [  0.00000000e+00,  -9.47850712e-15,   0.00000000e+00,
              1.00000000e+00]])
    >>> is_orthonormal_cont(polys, -1, 1, atol=1e-6)
    True

    """
    for i in range(len(polys)):
        for j in range(i + 1):
            p1 = polys[i]
            p2 = polys[j]
            innerprod = integrate.quad(lambda x: p1(x) * p2(x), lower, upper)[0]
            if not np.allclose(innerprod, i == j, rtol=rtol, atol=atol):
                return False
    return True