from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
def combine_indices(groups, prefix='', sep='.', return_labels=False):
    """use np.unique to get integer group indices for product, intersection
    """
    if isinstance(groups, tuple):
        groups = np.column_stack(groups)
    else:
        groups = np.asarray(groups)
    dt = groups.dtype
    is2d = groups.ndim == 2
    if is2d:
        ncols = groups.shape[1]
        if not groups.flags.c_contiguous:
            groups = np.array(groups, order='C')
        groups_ = groups.view([('', groups.dtype)] * groups.shape[1])
    else:
        groups_ = groups
    uni, uni_idx, uni_inv = np.unique(groups_, return_index=True, return_inverse=True)
    if is2d:
        uni = uni.view(dt).reshape(-1, ncols)
    if return_labels:
        label = [(prefix + sep.join(['%s'] * len(uni[0]))) % tuple(ii) for ii in uni]
        return (uni_inv, uni_idx, uni, label)
    else:
        return (uni_inv, uni_idx, uni)