from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
def test_intersection_equal_sort_true(self):
    idx = Index(['c', 'a', 'b'])
    sorted_ = Index(['a', 'b', 'c'])
    tm.assert_index_equal(idx.intersection(idx, sort=True), sorted_)