from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.grouper import Grouping
def test_empty_groups(self, df):
    with pytest.raises(ValueError, match='No group keys passed!'):
        df.groupby([])