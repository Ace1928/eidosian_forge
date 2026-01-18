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
@pytest.fixture(params=[Index, Series, tm.to_array, np.array, list], ids=lambda x: x.__name__)
def box_1d_array(request):
    """
    Fixture to test behavior for Index, Series, tm.to_array, numpy Array and list
    classes
    """
    return request.param