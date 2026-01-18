from statsmodels.compat.python import lrange, lzip
import numpy as np
import pandas as pd
import statsmodels.tools.data as data_util
from pandas import Index, MultiIndex
@property
def index_shape(self):
    if hasattr(self.index, 'levshape'):
        return self.index.levshape
    else:
        return self.index.shape