import functools
from threading import RLock
import numpy as np
from scipy.optimize import _cobyla as cobyla
from ._optimize import (OptimizeResult, _check_unknown_options,
def lb_constraint(x, *args, **kwargs):
    return x[i_lb] - bounds.lb[i_lb]