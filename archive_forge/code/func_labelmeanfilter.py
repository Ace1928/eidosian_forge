from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage
def labelmeanfilter(y, x):
    labelsunique = np.arange(np.max(y) + 1)
    labelmeans = np.array(ndimage.mean(x, labels=y, index=labelsunique))
    return labelmeans[y]