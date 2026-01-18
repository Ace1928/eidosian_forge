import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.parametrize('limit_area, input_ilocs, expected_ilocs', [('outside', [1, 0, 0, 0, 1], [1, 0, 0, 0, 1]), ('outside', [1, 0, 1, 0, 1], [1, 0, 1, 0, 1]), ('outside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 1]), ('outside', [0, 1, 0, 1, 0], [0, 1, 0, 1, 1]), ('inside', [1, 0, 0, 0, 1], [1, 1, 1, 1, 1]), ('inside', [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]), ('inside', [0, 1, 1, 1, 0], [0, 1, 1, 1, 0]), ('inside', [0, 1, 0, 1, 0], [0, 1, 1, 1, 0])])
def test_ffill_limit_area(self, data_missing, limit_area, input_ilocs, expected_ilocs):
    msg = 'JSONArray does not implement limit_area'
    with pytest.raises(NotImplementedError, match=msg):
        super().test_ffill_limit_area(data_missing, limit_area, input_ilocs, expected_ilocs)