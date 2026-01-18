from statsmodels.compat.python import lrange
import numpy as np
from scipy import stats
from statsmodels.sandbox.tools.mctools import StatTestMC
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
def randwalksim(nobs=500, drift=0.0):
    return (drift + np.random.randn(nobs)).cumsum()