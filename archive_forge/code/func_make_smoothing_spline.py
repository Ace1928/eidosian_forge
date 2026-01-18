import operator
from math import prod
import numpy as np
from scipy._lib._util import normalize_axis_index
from scipy.linalg import (get_lapack_funcs, LinAlgError,
from scipy.optimize import minimize_scalar
from . import _bspl
from . import _fitpack_impl
from scipy.sparse import csr_array
from scipy.special import poch
from itertools import combinations
def make_smoothing_spline(x, y, w=None, lam=None):
    """
    Compute the (coefficients of) smoothing cubic spline function using
    ``lam`` to control the tradeoff between the amount of smoothness of the
    curve and its proximity to the data. In case ``lam`` is None, using the
    GCV criteria [1] to find it.

    A smoothing spline is found as a solution to the regularized weighted
    linear regression problem:

    .. math::

        \\sum\\limits_{i=1}^n w_i\\lvert y_i - f(x_i) \\rvert^2 +
        \\lambda\\int\\limits_{x_1}^{x_n} (f^{(2)}(u))^2 d u

    where :math:`f` is a spline function, :math:`w` is a vector of weights and
    :math:`\\lambda` is a regularization parameter.

    If ``lam`` is None, we use the GCV criteria to find an optimal
    regularization parameter, otherwise we solve the regularized weighted
    linear regression problem with given parameter. The parameter controls
    the tradeoff in the following way: the larger the parameter becomes, the
    smoother the function gets.

    Parameters
    ----------
    x : array_like, shape (n,)
        Abscissas. `n` must be at least 5.
    y : array_like, shape (n,)
        Ordinates. `n` must be at least 5.
    w : array_like, shape (n,), optional
        Vector of weights. Default is ``np.ones_like(x)``.
    lam : float, (:math:`\\lambda \\geq 0`), optional
        Regularization parameter. If ``lam`` is None, then it is found from
        the GCV criteria. Default is None.

    Returns
    -------
    func : a BSpline object.
        A callable representing a spline in the B-spline basis
        as a solution of the problem of smoothing splines using
        the GCV criteria [1] in case ``lam`` is None, otherwise using the
        given parameter ``lam``.

    Notes
    -----
    This algorithm is a clean room reimplementation of the algorithm
    introduced by Woltring in FORTRAN [2]. The original version cannot be used
    in SciPy source code because of the license issues. The details of the
    reimplementation are discussed here (available only in Russian) [4].

    If the vector of weights ``w`` is None, we assume that all the points are
    equal in terms of weights, and vector of weights is vector of ones.

    Note that in weighted residual sum of squares, weights are not squared:
    :math:`\\sum\\limits_{i=1}^n w_i\\lvert y_i - f(x_i) \\rvert^2` while in
    ``splrep`` the sum is built from the squared weights.

    In cases when the initial problem is ill-posed (for example, the product
    :math:`X^T W X` where :math:`X` is a design matrix is not a positive
    defined matrix) a ValueError is raised.

    References
    ----------
    .. [1] G. Wahba, "Estimating the smoothing parameter" in Spline models for
        observational data, Philadelphia, Pennsylvania: Society for Industrial
        and Applied Mathematics, 1990, pp. 45-65.
        :doi:`10.1137/1.9781611970128`
    .. [2] H. J. Woltring, A Fortran package for generalized, cross-validatory
        spline smoothing and differentiation, Advances in Engineering
        Software, vol. 8, no. 2, pp. 104-113, 1986.
        :doi:`10.1016/0141-1195(86)90098-7`
    .. [3] T. Hastie, J. Friedman, and R. Tisbshirani, "Smoothing Splines" in
        The elements of Statistical Learning: Data Mining, Inference, and
        prediction, New York: Springer, 2017, pp. 241-249.
        :doi:`10.1007/978-0-387-84858-7`
    .. [4] E. Zemlyanoy, "Generalized cross-validation smoothing splines",
        BSc thesis, 2022.
        `<https://www.hse.ru/ba/am/students/diplomas/620910604>`_ (in
        Russian)

    Examples
    --------
    Generate some noisy data

    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> n = 200
    >>> def func(x):
    ...    return x**3 + x**2 * np.sin(4 * x)
    >>> x = np.sort(np.random.random_sample(n) * 4 - 2)
    >>> y = func(x) + np.random.normal(scale=1.5, size=n)

    Make a smoothing spline function

    >>> from scipy.interpolate import make_smoothing_spline
    >>> spl = make_smoothing_spline(x, y)

    Plot both

    >>> import matplotlib.pyplot as plt
    >>> grid = np.linspace(x[0], x[-1], 400)
    >>> plt.plot(grid, spl(grid), label='Spline')
    >>> plt.plot(grid, func(grid), label='Original function')
    >>> plt.scatter(x, y, marker='.')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    if any(x[1:] - x[:-1] <= 0):
        raise ValueError('``x`` should be an ascending array')
    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError('``x`` and ``y`` should be one dimensional and the same size')
    if w is None:
        w = np.ones(len(x))
    else:
        w = np.ascontiguousarray(w)
        if any(w <= 0):
            raise ValueError('Invalid vector of weights')
    t = np.r_[[x[0]] * 3, x, [x[-1]] * 3]
    n = x.shape[0]
    if n <= 4:
        raise ValueError('``x`` and ``y`` length must be at least 5')
    X_bspl = BSpline.design_matrix(x, t, 3)
    X = np.zeros((5, n))
    for i in range(1, 4):
        X[i, 2:-2] = X_bspl[i:i - 4, 3:-3][np.diag_indices(n - 4)]
    X[1, 1] = X_bspl[0, 0]
    X[2, :2] = ((x[2] + x[1] - 2 * x[0]) * X_bspl[0, 0], X_bspl[1, 1] + X_bspl[1, 2])
    X[3, :2] = ((x[2] - x[0]) * X_bspl[1, 1], X_bspl[2, 2])
    X[1, -2:] = (X_bspl[-3, -3], (x[-1] - x[-3]) * X_bspl[-2, -2])
    X[2, -2:] = (X_bspl[-2, -3] + X_bspl[-2, -2], (2 * x[-1] - x[-2] - x[-3]) * X_bspl[-1, -1])
    X[3, -2] = X_bspl[-1, -1]
    wE = np.zeros((5, n))
    wE[2:, 0] = _coeff_of_divided_diff(x[:3]) / w[:3]
    wE[1:, 1] = _coeff_of_divided_diff(x[:4]) / w[:4]
    for j in range(2, n - 2):
        wE[:, j] = (x[j + 2] - x[j - 2]) * _coeff_of_divided_diff(x[j - 2:j + 3]) / w[j - 2:j + 3]
    wE[:-1, -2] = -_coeff_of_divided_diff(x[-4:]) / w[-4:]
    wE[:-2, -1] = _coeff_of_divided_diff(x[-3:]) / w[-3:]
    wE *= 6
    if lam is None:
        lam = _compute_optimal_gcv_parameter(X, wE, y, w)
    elif lam < 0.0:
        raise ValueError('Regularization parameter should be non-negative')
    c = solve_banded((2, 2), X + lam * wE, y)
    c_ = np.r_[c[0] * (t[5] + t[4] - 2 * t[3]) + c[1], c[0] * (t[5] - t[3]) + c[1], c[1:-1], c[-1] * (t[-4] - t[-6]) + c[-2], c[-1] * (2 * t[-4] - t[-5] - t[-6]) + c[-2]]
    return BSpline.construct_fast(t, c_, 3)