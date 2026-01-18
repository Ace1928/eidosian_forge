from statsmodels.compat.python import lrange
import scipy
import numpy as np
from itertools import product
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Autoregressive, Nested
def gendat_nested1():
    ns = Nested_simulator()
    ns.error_sd = 2.0
    ns.params = np.r_[0, 1, 1.3, -0.8, -1.2]
    ns.ngroups = 50
    ns.nest_sizes = [10, 5]
    ns.dparams = [1.0, 3.0]
    ns.simulate()
    return (ns, Nested(ns.id_matrix))