from __future__ import annotations
import string
from typing import cast
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.types import is_string_dtype
from pandas.core.arrays import ArrowStringArray
from pandas.core.arrays.string_ import StringDtype
from pandas.tests.extension import base
@pytest.fixture(autouse=True)
def arrow_not_supported(self, data):
    if isinstance(data, ArrowStringArray):
        pytest.skip(reason='2D support not implemented for ArrowStringArray')