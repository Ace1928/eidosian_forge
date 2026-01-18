import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_groupby_corner(self):
    midx = MultiIndex(levels=[['foo'], ['bar'], ['baz']], codes=[[0], [0], [0]], names=['one', 'two', 'three'])
    df = DataFrame([np.random.default_rng(2).random(4)], columns=['a', 'b', 'c', 'd'], index=midx)
    df.groupby(level='three')