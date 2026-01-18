import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def empty_index(dtype='int64', closed='right'):
    return IntervalIndex(np.array([], dtype=dtype), closed=closed)