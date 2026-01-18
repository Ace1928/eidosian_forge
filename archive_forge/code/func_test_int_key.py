from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_int_key(self, obj, key, expected, warn, val, indexer_sli, is_inplace):
    if not isinstance(key, int):
        pytest.skip('Not relevant for int key')
    with tm.assert_produces_warning(warn, match='incompatible dtype'):
        self.check_indexer(obj, key, expected, val, indexer_sli, is_inplace)
    if indexer_sli is tm.loc:
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, key, expected, val, tm.at, is_inplace)
    elif indexer_sli is tm.iloc:
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, key, expected, val, tm.iat, is_inplace)
    rng = range(key, key + 1)
    with tm.assert_produces_warning(warn, match='incompatible dtype'):
        self.check_indexer(obj, rng, expected, val, indexer_sli, is_inplace)
    if indexer_sli is not tm.loc:
        slc = slice(key, key + 1)
        with tm.assert_produces_warning(warn, match='incompatible dtype'):
            self.check_indexer(obj, slc, expected, val, indexer_sli, is_inplace)
    ilkey = [key]
    with tm.assert_produces_warning(warn, match='incompatible dtype'):
        self.check_indexer(obj, ilkey, expected, val, indexer_sli, is_inplace)
    indkey = np.array(ilkey)
    with tm.assert_produces_warning(warn, match='incompatible dtype'):
        self.check_indexer(obj, indkey, expected, val, indexer_sli, is_inplace)
    genkey = (x for x in [key])
    with tm.assert_produces_warning(warn, match='incompatible dtype'):
        self.check_indexer(obj, genkey, expected, val, indexer_sli, is_inplace)