import numpy as np
from scipy import integrate, stats, special
from scipy.stats import chi
from .extras import mvstdnormcdf
from numpy import exp as np_exp
from numpy import log as np_log
from scipy.special import gamma as sps_gamma
from scipy.special import gammaln as sps_gammaln
def funbgh(s, a, b, R, df):
    sqrt_df = np.sqrt(df + 0.5)
    ret = chi_logpdf(s, df)
    ret += np_log(mvstdnormcdf(s * a / sqrt_df, s * b / sqrt_df, R, maxpts=1000000, abseps=1e-06))
    ret = np_exp(ret)
    return ret