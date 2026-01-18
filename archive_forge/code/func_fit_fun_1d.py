import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def fit_fun_1d(data):
    return dist.fit(data, *guessed_shapes, **guessed_params, **fixed_params)