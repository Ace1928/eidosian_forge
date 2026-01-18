from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage
def cat2dummy(y, nonseq=0):
    if nonseq or (y.ndim == 2 and y.shape[1] > 1):
        ycat, uniques, unitransl = convertlabels(y, lrange(y.shape[1]))
    else:
        ycat = y.copy()
        ymin = y.min()
        uniques = np.arange(ymin, y.max() + 1)
    if ycat.ndim == 1:
        ycat = ycat[:, np.newaxis]
    dummy = (ycat == uniques).astype(int)
    return dummy