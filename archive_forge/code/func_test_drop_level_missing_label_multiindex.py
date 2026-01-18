import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_level_missing_label_multiindex(self):
    df = DataFrame(index=MultiIndex.from_product([range(3), range(3)]))
    with pytest.raises(KeyError, match='labels \\[5\\] not found in level'):
        df.drop(5, level=0)