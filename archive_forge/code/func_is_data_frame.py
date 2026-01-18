from statsmodels.compat.numpy import NP_LT_2
import numpy as np
import pandas as pd
def is_data_frame(obj):
    return isinstance(obj, pd.DataFrame)