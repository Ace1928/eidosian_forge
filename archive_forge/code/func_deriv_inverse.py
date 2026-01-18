import warnings
import numpy as np
from scipy.special import expm1, gamma
def deriv_inverse(self, phi, theta):
    return -(1 + phi) ** (-(theta + 1) / theta) / theta