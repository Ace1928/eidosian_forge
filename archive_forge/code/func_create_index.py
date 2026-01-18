import numpy as np
import pytest
from pandas import IntervalIndex
import pandas._testing as tm
from pandas.tests.indexes.common import Base
def create_index(self, *, closed='right'):
    return IntervalIndex.from_breaks(range(11), closed=closed)