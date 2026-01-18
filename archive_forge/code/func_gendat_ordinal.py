from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.genmod.generalized_estimating_equations import GEE,\
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import GlobalOddsRatio
from .gee_gaussian_simulation_check import GEE_simulator
def gendat_ordinal():
    os = ordinal_simulator()
    os.params = np.r_[0.0, 1]
    os.ngroups = 200
    os.thresholds = [1, 0, -1]
    os.dparams = [1.0]
    os.simulate()
    data = np.concatenate((os.endog[:, None], os.exog, os.group[:, None]), axis=1)
    os.endog_ex, os.exog_ex, os.intercepts, os.nthresh = gee_setup_ordinal(data, 0)
    os.group_ex = os.exog_ex[:, -1]
    os.exog_ex = os.exog_ex[:, 0:-1]
    os.exog_ex = np.concatenate((os.intercepts, os.exog_ex), axis=1)
    va = GlobalOddsRatio(4, 'ordinal')
    lhs = np.array([[0.0, 0.0, 0, 1.0, 0.0], [0.0, 0, 0, 0, 1]])
    rhs = np.r_[0.0, 1]
    return (os, va, Binomial(), (lhs, rhs))