from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def is_SparseSeries(X):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'The SparseSeries class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version', FutureWarning)
        try:
            return isinstance(X, pd.SparseSeries)
        except AttributeError:
            return False