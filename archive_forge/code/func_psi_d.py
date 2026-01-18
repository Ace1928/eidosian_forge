import sys
import numpy as np
from scipy import stats, integrate, optimize
from . import transforms
from .copulas import Copula
from statsmodels.tools.rng_qrng import check_random_state
def psi_d(*args):
    return self.transform.derivk_inverse(k, *args)