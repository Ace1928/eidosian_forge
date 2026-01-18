from __future__ import annotations
from collections import abc
from datetime import timedelta
from decimal import Decimal
import operator
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.tests.arithmetic.common import (
@pytest.fixture(params=[Index, Series, tm.to_array])
def box_pandas_1d_array(request):
    """
    Fixture to test behavior for Index, Series and tm.to_array classes
    """
    return request.param