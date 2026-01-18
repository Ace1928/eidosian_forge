import warnings
import numpy as np
from scipy.special import expm1, gamma
def derivk_inverse(self, k, phi, theta):
    thi = 1 / theta
    d4 = (-1) ** k * gamma(k + thi) / gamma(thi) * (1 + phi) ** (-(k + thi))
    return d4