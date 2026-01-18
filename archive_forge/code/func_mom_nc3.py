import numpy as np
from scipy import stats
from statsmodels.sandbox.distributions.sppatch import expect_v2
from .distparams import distcont
def mom_nc3(x):
    return x * x * x