from __future__ import annotations
import pytest
import pandas as pd
from pandas import api
import pandas._testing as tm
from pandas.api import (
def test_api_all(self):
    expected = set(self.public_lib + self.misc + self.modules + self.classes + self.funcs + self.funcs_option + self.funcs_read + self.funcs_json + self.funcs_to) - set(self.deprecated_classes)
    actual = set(pd.__all__)
    extraneous = actual - expected
    assert not extraneous
    missing = expected - actual
    assert not missing