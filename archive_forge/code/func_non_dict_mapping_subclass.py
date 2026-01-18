from __future__ import annotations
from collections import abc
from datetime import (
from decimal import Decimal
import operator
import os
from typing import (
from dateutil.tz import (
import hypothesis
from hypothesis import strategies as st
import numpy as np
import pytest
from pytz import (
from pandas._config.config import _get_option
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.indexes.api import (
from pandas.util.version import Version
import zoneinfo
@pytest.fixture
def non_dict_mapping_subclass() -> type[abc.Mapping]:
    """
    Fixture for a non-mapping dictionary subclass.
    """

    class TestNonDictMapping(abc.Mapping):

        def __init__(self, underlying_dict) -> None:
            self._data = underlying_dict

        def __getitem__(self, key):
            return self._data.__getitem__(key)

        def __iter__(self) -> Iterator:
            return self._data.__iter__()

        def __len__(self) -> int:
            return self._data.__len__()
    return TestNonDictMapping