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
def test_from_sequence_of_strings_boolean():
    true_strings = ['true', 'TRUE', 'True', '1', '1.0']
    false_strings = ['false', 'FALSE', 'False', '0', '0.0']
    nulls = [None]
    strings = true_strings + false_strings + nulls
    bools = [True] * len(true_strings) + [False] * len(false_strings) + [None] * len(nulls)
    result = ArrowExtensionArray._from_sequence_of_strings(strings, dtype=pa.bool_())
    expected = pd.array(bools, dtype='boolean[pyarrow]')
    tm.assert_extension_array_equal(result, expected)
    strings = ['True', 'foo']
    with pytest.raises(pa.ArrowInvalid, match='Failed to parse'):
        ArrowExtensionArray._from_sequence_of_strings(strings, dtype=pa.bool_())