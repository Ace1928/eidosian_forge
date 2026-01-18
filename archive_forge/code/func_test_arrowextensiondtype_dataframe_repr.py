from __future__ import annotations
from datetime import (
from decimal import Decimal
from io import (
import operator
import pickle
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import timezones
from pandas.compat import (
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import (
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import no_default
from pandas.api.types import (
from pandas.tests.extension import base
from pandas.core.arrays.arrow.array import ArrowExtensionArray
from pandas.core.arrays.arrow.extension_types import ArrowPeriodType
def test_arrowextensiondtype_dataframe_repr():
    df = pd.DataFrame(pd.period_range('2012', periods=3), columns=['col'], dtype=ArrowDtype(ArrowPeriodType('D')))
    result = repr(df)
    expected = '     col\n0  15340\n1  15341\n2  15342'
    assert result == expected