from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
def gen_data(dim, use_pandas):
    if dim == 1:
        out = np.empty(10)
        if use_pandas:
            out = pd.Series(out)
    elif dim == 2:
        out = np.empty((20, 10))
        if use_pandas:
            out = pd.DataFrame(out)
    else:
        out = np.empty(np.arange(5, 5 + dim))
    return out