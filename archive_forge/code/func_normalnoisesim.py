from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.sandbox.tools.mctools import StatTestMC
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
def normalnoisesim(nobs=500, loc=0.0):
    return loc + np.random.randn(nobs)