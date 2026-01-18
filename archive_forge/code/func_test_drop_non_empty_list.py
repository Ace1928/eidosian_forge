import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', [[1, 2, 3], [1, 2, 2]])
@pytest.mark.parametrize('drop_labels', [[1, 4], [4, 5]])
def test_drop_non_empty_list(self, index, drop_labels):
    with pytest.raises(KeyError, match='not found in axis'):
        DataFrame(index=index).drop(drop_labels)