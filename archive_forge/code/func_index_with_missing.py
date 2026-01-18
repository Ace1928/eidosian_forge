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
@pytest.fixture(params=[key for key, value in indices_dict.items() if not (key.startswith(('int', 'uint', 'float')) or key in ['range', 'empty', 'repeats', 'bool-dtype']) and (not isinstance(value, MultiIndex))])
def index_with_missing(request):
    """
    Fixture for indices with missing values.

    Integer-dtype and empty cases are excluded because they cannot hold missing
    values.

    MultiIndex is excluded because isna() is not defined for MultiIndex.
    """
    ind = indices_dict[request.param].copy(deep=True)
    vals = ind.values.copy()
    if request.param in ['tuples', 'mi-with-dt64tz-level', 'multi']:
        vals = ind.tolist()
        vals[0] = (None,) + vals[0][1:]
        vals[-1] = (None,) + vals[-1][1:]
        return MultiIndex.from_tuples(vals)
    else:
        vals[0] = None
        vals[-1] = None
        return type(ind)(vals)