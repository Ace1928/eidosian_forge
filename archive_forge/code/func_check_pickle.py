from statsmodels.compat.pandas import assert_series_equal
from io import BytesIO
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
def check_pickle(obj):
    fh = BytesIO()
    pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    plen = fh.tell()
    fh.seek(0, 0)
    res = pickle.load(fh)
    fh.close()
    return (res, plen)