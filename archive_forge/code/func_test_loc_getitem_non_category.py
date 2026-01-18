import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_non_category(self, df2):
    with pytest.raises(KeyError, match=re.escape("['d'] not in index")):
        df2.loc[['a', 'd']]