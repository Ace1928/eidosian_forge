import numpy as np
from statsmodels.tools.validation import array_like
def bic_sigma(sigma2, nobs, df_modelwc, islog=False):
    """Bayesian information criterion (BIC) or Schwarz criterion

    Parameters
    ----------
    sigma2 : float
        estimate of the residual variance or determinant of Sigma_hat in the
        multivariate case. If islog is true, then it is assumed that sigma
        is already log-ed, for example logdetSigma.
    nobs : int
        number of observations
    df_modelwc : int
        number of parameters including constant

    Returns
    -------
    bic : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable
    models.

    References
    ----------
    https://en.wikipedia.org/wiki/Bayesian_information_criterion
    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + bic(0, nobs, df_modelwc) / nobs