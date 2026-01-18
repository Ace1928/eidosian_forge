from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def filt2(x):
    if x.shape[0] == 1:
        return x
    else:
        return x[x.category == 'c']