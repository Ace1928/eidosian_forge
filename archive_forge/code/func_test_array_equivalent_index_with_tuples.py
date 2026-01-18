from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_array_equivalent_index_with_tuples():
    idx1 = Index(np.array([(pd.NA, 4), (1, 1)], dtype='object'))
    idx2 = Index(np.array([(1, 1), (pd.NA, 4)], dtype='object'))
    assert not array_equivalent(idx1, idx2)
    assert not idx1.equals(idx2)
    assert not array_equivalent(idx2, idx1)
    assert not idx2.equals(idx1)
    idx1 = Index(np.array([(4, pd.NA), (1, 1)], dtype='object'))
    idx2 = Index(np.array([(1, 1), (4, pd.NA)], dtype='object'))
    assert not array_equivalent(idx1, idx2)
    assert not idx1.equals(idx2)
    assert not array_equivalent(idx2, idx1)
    assert not idx2.equals(idx1)