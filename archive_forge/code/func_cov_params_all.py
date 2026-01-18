import warnings
import numpy as np
from scipy import stats
from statsmodels.tools.decorators import cache_readonly
from statsmodels.regression.linear_model import OLS
@cache_readonly
def cov_params_all(self):
    m_deriv = np.zeros((self.k_moments_all, self.k_moments_all))
    m_deriv[:self.k_params, :self.k_params] = self.score_deriv
    m_deriv[self.k_params:, :self.k_params] = self.moments_deriv
    m_deriv[self.k_params:, self.k_params:] = np.eye(self.k_moments_test)
    m_deriv_inv = np.linalg.inv(m_deriv)
    cov = m_deriv_inv.dot(self.cov_moments_all.dot(m_deriv_inv.T))
    return cov