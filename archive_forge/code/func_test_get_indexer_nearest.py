from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer_nearest(self):
    midx = MultiIndex.from_tuples([('a', 1), ('b', 2)])
    msg = "method='nearest' not implemented yet for MultiIndex; see GitHub issue 9365"
    with pytest.raises(NotImplementedError, match=msg):
        midx.get_indexer(['a'], method='nearest')
    msg = 'tolerance not implemented yet for MultiIndex'
    with pytest.raises(NotImplementedError, match=msg):
        midx.get_indexer(['a'], method='pad', tolerance=2)