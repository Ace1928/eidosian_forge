from statsmodels.compat.python import lrange
import numpy as np
from scipy import ndimage
def labelmeanfilter_str(ys, x):
    unil, unilinv = np.unique(ys, return_index=False, return_inverse=True)
    labelmeans = np.array(ndimage.mean(x, labels=unilinv, index=np.arange(np.max(unil) + 1)))
    arr3 = labelmeans[unilinv]
    return arr3