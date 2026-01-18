from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def reindex_helper(x):
    return x.reindex(np.arange(x.index.min(), x.index.max() + 1))