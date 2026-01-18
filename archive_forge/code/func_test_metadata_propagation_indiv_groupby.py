from copy import deepcopy
from operator import methodcaller
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_metadata_propagation_indiv_groupby(self):
    df = DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'], 'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'], 'C': np.random.default_rng(2).standard_normal(8), 'D': np.random.default_rng(2).standard_normal(8)})
    result = df.groupby('A').sum()
    tm.assert_metadata_equivalent(df, result)