import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.tests.extension.base import BaseOpsUtil
def test_no_shared_mask(self, data):
    result = data + 1
    assert not tm.shares_memory(result, data)