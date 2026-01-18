from __future__ import annotations
import pytest
import pandas as pd
from pandas import api
import pandas._testing as tm
from pandas.api import (
def test_api_indexers(self):
    self.check(api_indexers, self.allowed_api_indexers)