from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def is_sparse_series(x):
    if isinstance(x, pd.Series) and (not is_SparseSeries(x)):
        try:
            x.sparse
            return True
        except AttributeError:
            pass
    return False