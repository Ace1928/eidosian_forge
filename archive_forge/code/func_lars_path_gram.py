import sys
import warnings
from math import log
from numbers import Integral, Real
import numpy as np
from scipy import interpolate, linalg
from scipy.linalg.lapack import get_lapack_funcs
from ..base import MultiOutputMixin, RegressorMixin, _fit_context
from ..exceptions import ConvergenceWarning
from ..model_selection import check_cv
from ..utils import (  # type: ignore
from ..utils._metadata_requests import (
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.parallel import Parallel, delayed
from ._base import LinearModel, LinearRegression, _preprocess_data
@validate_params({'Xy': [np.ndarray], 'Gram': [np.ndarray], 'n_samples': [Interval(Integral, 0, None, closed='left')], 'max_iter': [Interval(Integral, 0, None, closed='left')], 'alpha_min': [Interval(Real, 0, None, closed='left')], 'method': [StrOptions({'lar', 'lasso'})], 'copy_X': ['boolean'], 'eps': [Interval(Real, 0, None, closed='neither'), None], 'copy_Gram': ['boolean'], 'verbose': ['verbose'], 'return_path': ['boolean'], 'return_n_iter': ['boolean'], 'positive': ['boolean']}, prefer_skip_nested_validation=True)
def lars_path_gram(Xy, Gram, *, n_samples, max_iter=500, alpha_min=0, method='lar', copy_X=True, eps=np.finfo(float).eps, copy_Gram=True, verbose=0, return_path=True, return_n_iter=False, positive=False):
    """The lars_path in the sufficient stats mode [1].

    The optimization objective for the case method='lasso' is::

    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1

    in the case of method='lars', the objective function is only known in
    the form of an implicit equation (see discussion in [1])

    Read more in the :ref:`User Guide <least_angle_regression>`.

    Parameters
    ----------
    Xy : ndarray of shape (n_features,) or (n_features, n_targets)
        `Xy = X.T @ y`.

    Gram : ndarray of shape (n_features, n_features)
        `Gram = X.T @ X`.

    n_samples : int
        Equivalent size of sample.

    max_iter : int, default=500
        Maximum number of iterations to perform, set to infinity for no limit.

    alpha_min : float, default=0
        Minimum correlation along the path. It corresponds to the
        regularization parameter alpha parameter in the Lasso.

    method : {'lar', 'lasso'}, default='lar'
        Specifies the returned model. Select `'lar'` for Least Angle
        Regression, ``'lasso'`` for the Lasso.

    copy_X : bool, default=True
        If `False`, `X` is overwritten.

    eps : float, default=np.finfo(float).eps
        The machine-precision regularization in the computation of the
        Cholesky diagonal factors. Increase this for very ill-conditioned
        systems. Unlike the `tol` parameter in some iterative
        optimization-based algorithms, this parameter does not control
        the tolerance of the optimization.

    copy_Gram : bool, default=True
        If `False`, `Gram` is overwritten.

    verbose : int, default=0
        Controls output verbosity.

    return_path : bool, default=True
        If `return_path==True` returns the entire path, else returns only the
        last point of the path.

    return_n_iter : bool, default=False
        Whether to return the number of iterations.

    positive : bool, default=False
        Restrict coefficients to be >= 0.
        This option is only allowed with method 'lasso'. Note that the model
        coefficients will not converge to the ordinary-least-squares solution
        for small values of alpha. Only coefficients up to the smallest alpha
        value (`alphas_[alphas_ > 0.].min()` when `fit_path=True`) reached by
        the stepwise Lars-Lasso algorithm are typically in congruence with the
        solution of the coordinate descent lasso_path function.

    Returns
    -------
    alphas : ndarray of shape (n_alphas + 1,)
        Maximum of covariances (in absolute value) at each iteration.
        `n_alphas` is either `max_iter`, `n_features` or the
        number of nodes in the path with `alpha >= alpha_min`, whichever
        is smaller.

    active : ndarray of shape (n_alphas,)
        Indices of active variables at the end of the path.

    coefs : ndarray of shape (n_features, n_alphas + 1)
        Coefficients along the path.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is set
        to True.

    See Also
    --------
    lars_path_gram : Compute LARS path.
    lasso_path : Compute Lasso path with coordinate descent.
    LassoLars : Lasso model fit with Least Angle Regression a.k.a. Lars.
    Lars : Least Angle Regression model a.k.a. LAR.
    LassoLarsCV : Cross-validated Lasso, using the LARS algorithm.
    LarsCV : Cross-validated Least Angle Regression model.
    sklearn.decomposition.sparse_encode : Sparse coding.

    References
    ----------
    .. [1] "Least Angle Regression", Efron et al.
           http://statweb.stanford.edu/~tibs/ftp/lars.pdf

    .. [2] `Wikipedia entry on the Least-angle regression
           <https://en.wikipedia.org/wiki/Least-angle_regression>`_

    .. [3] `Wikipedia entry on the Lasso
           <https://en.wikipedia.org/wiki/Lasso_(statistics)>`_
    """
    return _lars_path_solver(X=None, y=None, Xy=Xy, Gram=Gram, n_samples=n_samples, max_iter=max_iter, alpha_min=alpha_min, method=method, copy_X=copy_X, eps=eps, copy_Gram=copy_Gram, verbose=verbose, return_path=return_path, return_n_iter=return_n_iter, positive=positive)