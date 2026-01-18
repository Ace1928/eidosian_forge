import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
from pandas.api.extensions import ExtensionArray
from pandas.core.internals.blocks import EABackedBlock
def test_from_sequence_from_cls(self, data):
    result = type(data)._from_sequence(data, dtype=data.dtype)
    tm.assert_extension_array_equal(result, data)
    data = data[:0]
    result = type(data)._from_sequence(data, dtype=data.dtype)
    tm.assert_extension_array_equal(result, data)