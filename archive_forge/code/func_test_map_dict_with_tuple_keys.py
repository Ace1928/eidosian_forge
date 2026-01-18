from collections import (
from decimal import Decimal
import math
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_map_dict_with_tuple_keys():
    """
    Due to new MultiIndex-ing behaviour in v0.14.0,
    dicts with tuple keys passed to map were being
    converted to a multi-index, preventing tuple values
    from being mapped properly.
    """
    df = DataFrame({'a': [(1,), (2,), (3, 4), (5, 6)]})
    label_mappings = {(1,): 'A', (2,): 'B', (3, 4): 'A', (5, 6): 'B'}
    df['labels'] = df['a'].map(label_mappings)
    df['expected_labels'] = Series(['A', 'B', 'A', 'B'], index=df.index)
    tm.assert_series_equal(df['labels'], df['expected_labels'], check_names=False)