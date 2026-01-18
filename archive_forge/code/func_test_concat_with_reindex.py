import itertools
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_concat_with_reindex(self, data):
    a = pd.DataFrame({'a': data[:5]})
    b = pd.DataFrame({'b': data[:5]})
    result = pd.concat([a, b], ignore_index=True)
    expected = pd.DataFrame({'a': data.take(list(range(5)) + [-1] * 5, allow_fill=True), 'b': data.take([-1] * 5 + list(range(5)), allow_fill=True)})
    tm.assert_frame_equal(result, expected)