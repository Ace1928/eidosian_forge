from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def struct_to_ndarray(arr):
    return arr.view((float, (len(arr.dtype.names),)), type=np.ndarray)