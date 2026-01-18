from __future__ import annotations
import pytest
import pandas as pd
from pandas import api
import pandas._testing as tm
from pandas.api import (
def test_depr(self):
    deprecated_list = self.deprecated_classes + self.deprecated_funcs + self.deprecated_funcs_in_future
    for depr in deprecated_list:
        with tm.assert_produces_warning(FutureWarning):
            _ = getattr(pd, depr)