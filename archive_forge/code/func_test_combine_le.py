import collections
import operator
import sys
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.json.array import (
@pytest.mark.xfail(reason='combine for JSONArray not supported')
def test_combine_le(self, data_repeated):
    super().test_combine_le(data_repeated)