from datetime import datetime
import numpy as np
import pytest
from pandas.errors import MergeError
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
@pytest.fixture
def frame_with_period_index():
    return DataFrame(data=np.arange(20).reshape(4, 5), columns=list('abcde'), index=period_range(start='2000', freq='Y', periods=4))