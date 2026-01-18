from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
def test_divmod_numeric(self):
    td = Timedelta(days=2, hours=6)
    result = divmod(td, 53 * 3600 * 1000000000.0)
    assert result[0] == Timedelta(1, unit='ns')
    assert isinstance(result[1], Timedelta)
    assert result[1] == Timedelta(hours=1)
    assert result
    result = divmod(td, np.nan)
    assert result[0] is NaT
    assert result[1] is NaT