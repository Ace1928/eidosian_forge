from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage
def groupsstats_1d(y, x, labelsunique):
    """use ndimage to get fast mean and variance"""
    labelmeans = np.array(ndimage.mean(x, labels=y, index=labelsunique))
    labelvars = np.array(ndimage.var(x, labels=y, index=labelsunique))
    return (labelmeans, labelvars)