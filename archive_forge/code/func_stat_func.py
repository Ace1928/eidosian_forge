import numpy as np
import warnings
from scipy import stats, optimize
from statsmodels.stats.base import HolderTuple
from statsmodels.stats.weightstats import _zstat_generic2
from statsmodels.stats._inference_tools import _mover_confint
def stat_func(x1, x2):
    rate1, rate2 = (x1 / n1, x2 / n2)
    stat = rate1 - rate2 - value
    stat /= np.sqrt(rate1 / n1 + rate2 / n2 + eps)
    return stat