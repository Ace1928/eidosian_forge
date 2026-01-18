import numpy as np
from scipy.special import erf
def gaussian_convolution(h, Xi, x):
    """ Calculates the Gaussian Convolution Kernel """
    return 1.0 / np.sqrt(4 * np.pi) * np.exp(-(Xi - x) ** 2 / (h ** 2 * 4.0))