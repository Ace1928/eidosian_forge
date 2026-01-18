from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
def scaled_sum(*args):
    if len(args) < 2:
        raise ValueError('The function needs two arguments')
    array, scale = args
    return array.sum() / scale