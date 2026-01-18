from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage
def convertlabels(ys, indices=None):
    """convert labels based on multiple variables or string labels to unique
    index labels 0,1,2,...,nk-1 where nk is the number of distinct labels
    """
    if indices is None:
        ylabel = ys
    else:
        idx = np.array(indices)
        if idx.size > 1 and ys.ndim == 2:
            ylabel = np.array(['@%s@' % ii[:2].tostring() for ii in ys])[:, np.newaxis]
        else:
            ylabel = ys
    unil, unilinv = np.unique(ylabel, return_index=False, return_inverse=True)
    return (unilinv, np.arange(len(unil)), unil)