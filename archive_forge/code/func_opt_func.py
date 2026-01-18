from statsmodels.compat.python import lrange
import math
import scipy.stats
import numpy as np
from scipy.optimize import fminbound
def opt_func(p, r, v):
    return np.squeeze(abs(_qsturng(p, r, v) - q))