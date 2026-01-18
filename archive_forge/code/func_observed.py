import numpy as np
import pandas as pd
from pandas.core.nanops import nanmean as pd_nanmean
from statsmodels.tools.validation import PandasWrapper, array_like
from statsmodels.tsa.stl._stl import STL
from statsmodels.tsa.filters.filtertools import convolution_filter
from statsmodels.tsa.stl.mstl import MSTL
from statsmodels.tsa.tsatools import freq_to_period
@property
def observed(self):
    """Observed data"""
    return self._observed