from statsmodels.compat.python import lzip
import numpy as np
from scipy.stats import norm
from statsmodels.tools.decorators import cache_readonly
def margeff_cov_params(model, params, exog, cov_params, at, derivative, dummy_ind, count_ind, method, J):
    """
    Computes the variance-covariance of marginal effects by the delta method.

    Parameters
    ----------
    model : model instance
        The model that returned the fitted results. Its pdf method is used
        for computing the Jacobian of discrete variables in dummy_ind and
        count_ind
    params : array_like
        estimated model parameters
    exog : array_like
        exogenous variables at which to calculate the derivative
    cov_params : array_like
        The variance-covariance of the parameters
    at : str
       Options are:

        - 'overall', The average of the marginal effects at each
          observation.
        - 'mean', The marginal effects at the mean of each regressor.
        - 'median', The marginal effects at the median of each regressor.
        - 'zero', The marginal effects at zero for each regressor.
        - 'all', The marginal effects at each observation.

        Only overall has any effect here.you

    derivative : function or array_like
        If a function, it returns the marginal effects of the model with
        respect to the exogenous variables evaluated at exog. Expected to be
        called derivative(params, exog). This will be numerically
        differentiated. Otherwise, it can be the Jacobian of the marginal
        effects with respect to the parameters.
    dummy_ind : array_like
        Indices of the columns of exog that contain dummy variables
    count_ind : array_like
        Indices of the columns of exog that contain count variables

    Notes
    -----
    For continuous regressors, the variance-covariance is given by

    Asy. Var[MargEff] = [d margeff / d params] V [d margeff / d params]'

    where V is the parameter variance-covariance.

    The outer Jacobians are computed via numerical differentiation if
    derivative is a function.
    """
    if callable(derivative):
        from statsmodels.tools.numdiff import approx_fprime_cs
        params = params.ravel('F')
        try:
            jacobian_mat = approx_fprime_cs(params, derivative, args=(exog, method))
        except TypeError:
            from statsmodels.tools.numdiff import approx_fprime
            jacobian_mat = approx_fprime(params, derivative, args=(exog, method))
        if at == 'overall':
            jacobian_mat = np.mean(jacobian_mat, axis=1)
        else:
            jacobian_mat = jacobian_mat.squeeze()
        if dummy_ind is not None:
            jacobian_mat = _margeff_cov_params_dummy(model, jacobian_mat, params, exog, dummy_ind, method, J)
        if count_ind is not None:
            jacobian_mat = _margeff_cov_params_count(model, jacobian_mat, params, exog, count_ind, method, J)
    else:
        jacobian_mat = derivative
    return np.dot(np.dot(jacobian_mat, cov_params), jacobian_mat.T)