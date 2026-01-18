from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.genmod.generalized_estimating_equations import GEE,\
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import GlobalOddsRatio
from .gee_gaussian_simulation_check import GEE_simulator
def gendat_nominal():
    ns = nominal_simulator()
    ns.params = [np.r_[0.0, 1], np.r_[-1.0, 0], np.r_[0.0, 0]]
    ns.ngroups = 200
    ns.dparams = [1.0]
    ns.simulate()
    data = np.concatenate((ns.endog[:, None], ns.exog, ns.group[:, None]), axis=1)
    ns.endog_ex, ns.exog_ex, ns.exog_ne, ns.nlevel = gee_setup_nominal(data, 0, [3])
    ns.group_ex = ns.exog_ne[:, 0]
    va = GlobalOddsRatio(3, 'nominal')
    lhs = np.array([[0.0, 1.0, 1, 0]])
    rhs = np.r_[0.0,]
    return (ns, va, Multinomial(3), (lhs, rhs))