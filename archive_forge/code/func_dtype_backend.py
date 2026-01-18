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
@pytest.fixture(params=['numpy_nullable', pytest.param('pyarrow', marks=td.skip_if_no('pyarrow'))])
def dtype_backend(request):
    """
    Parametrized fixture for pd.options.mode.string_storage.

    * 'python'
    * 'pyarrow'
    """
    return request.param