from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_contains_td64_level(self):
    tx = pd.timedelta_range('09:30:00', '16:00:00', freq='30 min')
    idx = MultiIndex.from_arrays([tx, np.arange(len(tx))])
    assert tx[0] in idx
    assert 'element_not_exit' not in idx
    assert '0 day 09:30:00' in idx