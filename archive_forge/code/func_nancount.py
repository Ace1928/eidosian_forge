from statsmodels.compat.pandas import PD_LT_2, Appender, is_numeric_dtype
from statsmodels.compat.scipy import SP_LT_19
from typing import Union
from collections.abc import Sequence
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.stattools import jarque_bera
from statsmodels.tools.decorators import cache_readonly
from statsmodels.tools.docstring import Docstring, Parameter
from statsmodels.tools.validation import (
def nancount(x, axis=0):
    return (1 - np.isnan(x)).sum(axis=axis)