from decorator import decorator
from scipy import sparse
import importlib
import numbers
import numpy as np
import pandas as pd
import re
import warnings
def is_SparseDataFrame(X):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version', FutureWarning)
        try:
            return isinstance(X, pd.SparseDataFrame)
        except AttributeError:
            return False