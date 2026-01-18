from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_loc_nested_tuple_raises_keyerror(self):
    mi = MultiIndex.from_product([range(3), range(4), range(5), range(6)])
    key = ((2, 3, 4), 'foo')
    with pytest.raises(KeyError, match=re.escape(str(key))):
        mi.get_loc(key)