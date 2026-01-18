import numpy as np
from statsmodels.tools.validation import array_like
def aicc_sigma(sigma2, nobs, df_modelwc, islog=False):
    """
    Akaike information criterion (AIC) with small sample correction

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
    aicc : float
        information criterion

    Notes
    -----
    A constant has been dropped in comparison to the loglikelihood base
    information criteria. These should be used to compare for comparable
    models.

    References
    ----------
    https://en.wikipedia.org/wiki/Akaike_information_criterion#AICc
    """
    if not islog:
        sigma2 = np.log(sigma2)
    return sigma2 + aicc(0, nobs, df_modelwc) / nobs