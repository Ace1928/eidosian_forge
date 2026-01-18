from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def create_index_with_nan(self, closed='right'):
    mask = [True, False] + [True] * 8
    return IntervalIndex.from_arrays(np.where(mask, np.arange(10), np.nan), np.where(mask, np.arange(1, 11), np.nan), closed=closed)