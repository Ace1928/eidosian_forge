from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_join_self_unique(self, join_type, simple_index):
    idx = simple_index
    if idx.is_unique:
        joined = idx.join(idx, how=join_type)
        assert (idx == joined).all()