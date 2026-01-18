import numpy as np
from statsmodels.genmod.families import Poisson
from .gee_gaussian_simulation_check import GEE_simulator
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable,Independence
def gendat_overdispersed():
    exs = Overdispersed_simulator()
    exs.params = np.r_[2.0, 0.2, 0.2, -0.1, -0.2]
    exs.ngroups = 200
    exs.scale_inv = 2.0
    exs.dparams = []
    exs.simulate()
    return (exs, Independence())