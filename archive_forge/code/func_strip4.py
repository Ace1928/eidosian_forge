from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import (
from collections import namedtuple
import numpy as np
from pandas import DataFrame, MultiIndex, Series
from scipy import stats
from statsmodels.base import model
from statsmodels.base.model import LikelihoodModelResults, Model
from statsmodels.regression.linear_model import (
from statsmodels.tools.validation import array_like, int_like, string_like
def strip4(line):
    if line.startswith(' '):
        return line[4:]
    return line