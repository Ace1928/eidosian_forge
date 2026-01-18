from datetime import timezone
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_missing_axis_specification_exception(self):
    df = DataFrame(np.arange(50).reshape((10, 5)))
    series = Series(np.arange(5))
    with pytest.raises(ValueError, match='axis=0 or 1'):
        df.align(series)