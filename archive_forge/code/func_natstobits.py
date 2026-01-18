from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def natstobits(X):
    """
    Converts from nats to bits
    """
    return logbasechange(np.e, 2) * X