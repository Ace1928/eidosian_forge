import numpy as np
from scipy import stats
import pandas as pd
from numpy.testing import assert_, assert_almost_equal, assert_allclose
from statsmodels.stats.weightstats import (DescrStatsW, CompareMeans,
from statsmodels.tools.testing import Holder
@classmethod
def get_descriptives(cls, ddof=0):
    cls.descriptive = DescrStatsW(cls.data, cls.weights, ddof)