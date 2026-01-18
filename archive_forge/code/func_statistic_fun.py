import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def statistic_fun(data, axis=-1):
    data = np.moveaxis(data, axis, -1)
    rfd_vals = fit_fun(data)
    rfd_dist = dist(*rfd_vals)
    return compare_fun(rfd_dist, data)