from statsmodels.compat.python import lrange
import scipy
import numpy as np
from itertools import product
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Autoregressive, Nested
def check_constraint(da, va, ga):
    """
    Check the score testing of the parameter constraints.
    """