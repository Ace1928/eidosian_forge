from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage
def groupsstats_dummy(y, x, nonseq=0):
    if x.ndim == 1:
        x = x[:, np.newaxis]
    dummy = cat2dummy(y, nonseq=nonseq)
    countgr = dummy.sum(0, dtype=float)
    meangr = np.dot(x.T, dummy) / countgr
    meandata = np.dot(dummy, meangr.T)
    xdevmeangr = x - meandata
    vargr = np.dot((xdevmeangr * xdevmeangr).T, dummy) / countgr
    return (meangr, vargr, xdevmeangr, countgr)