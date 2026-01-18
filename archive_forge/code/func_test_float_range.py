from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('args', [(), (0, -1)])
def test_float_range(self, args):
    str(Series(np.random.randn(1000), index=np.arange(1000, *args)))