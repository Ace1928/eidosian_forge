import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import (ValueWarning,
from statsmodels.tools.validation import (string_like,
def keep_col(x):
    index = np.logical_not(np.any(np.isnan(x), 0))
    return (x[:, index], index)