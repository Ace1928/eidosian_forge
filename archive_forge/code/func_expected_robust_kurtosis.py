from scipy import stats
import numpy as np
from statsmodels.tools.sm_exceptions import ValueWarning
def expected_robust_kurtosis(ab=(5.0, 50.0), dg=(2.5, 25.0)):
    """
    Calculates the expected value of the robust kurtosis measures in Kim and
    White assuming the data are normally distributed.

    Parameters
    ----------
    ab : iterable, optional
        Contains 100*(alpha, beta) in the kr3 measure where alpha is the tail
        quantile cut-off for measuring the extreme tail and beta is the central
        quantile cutoff for the standardization of the measure
    db : iterable, optional
        Contains 100*(delta, gamma) in the kr4 measure where delta is the tail
        quantile for measuring extreme values and gamma is the central quantile
        used in the the standardization of the measure

    Returns
    -------
    ekr : ndarray, 4-element
        Contains the expected values of the 4 robust kurtosis measures

    Notes
    -----
    See `robust_kurtosis` for definitions of the robust kurtosis measures
    """
    alpha, beta = ab
    delta, gamma = dg
    expected_value = np.zeros(4)
    ppf = stats.norm.ppf
    pdf = stats.norm.pdf
    q1, q2, q3, q5, q6, q7 = ppf(np.array((1.0, 2.0, 3.0, 5.0, 6.0, 7.0)) / 8)
    expected_value[0] = 3
    expected_value[1] = (q7 - q5 + (q3 - q1)) / (q6 - q2)
    q_alpha, q_beta = ppf(np.array((alpha / 100.0, beta / 100.0)))
    expected_value[2] = 2 * pdf(q_alpha) / alpha / (2 * pdf(q_beta) / beta)
    q_delta, q_gamma = ppf(np.array((delta / 100.0, gamma / 100.0)))
    expected_value[3] = -2.0 * q_delta / (-2.0 * q_gamma)
    return expected_value