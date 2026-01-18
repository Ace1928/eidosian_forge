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
@pytest.fixture(params=['object', 'string[python]', pytest.param('string[pyarrow]', marks=td.skip_if_no('pyarrow')), pytest.param('string[pyarrow_numpy]', marks=td.skip_if_no('pyarrow'))])
def any_string_dtype(request):
    """
    Parametrized fixture for string dtypes.
    * 'object'
    * 'string[python]'
    * 'string[pyarrow]'
    """
    return request.param