import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import (
from pandas.core.indexers.objects import (
from pandas.tseries.offsets import BusinessDay
def test_bad_get_window_bounds_signature():

    class BadIndexer(BaseIndexer):

        def get_window_bounds(self):
            return None
    indexer = BadIndexer()
    with pytest.raises(ValueError, match='BadIndexer does not implement'):
        Series(range(5)).rolling(indexer)