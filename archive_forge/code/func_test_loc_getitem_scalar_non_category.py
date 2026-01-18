import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_scalar_non_category(self, df):
    with pytest.raises(KeyError, match='^1$'):
        df.loc[1]