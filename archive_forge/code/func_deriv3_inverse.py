import warnings
import numpy as np
from scipy.special import expm1, gamma
def deriv3_inverse(self, phi, *args):
    return -np.exp(-phi)