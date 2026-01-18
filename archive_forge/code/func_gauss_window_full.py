import numpy as np
import scipy.fftpack
import scipy.io
import scipy.ndimage
import scipy.stats
from ..utils import *
def gauss_window_full(lw, sigma):
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * lw
    sd *= sd
    center = (lw - 1) / 2.0
    for ii in range(lw):
        x = ii - center
        tmp = np.exp(-0.5 * np.float32(x * x) / sd)
        weights[ii] = tmp
    weights /= np.sum(weights)
    return weights