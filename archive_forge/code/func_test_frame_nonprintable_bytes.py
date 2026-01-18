import datetime
from datetime import timedelta
from decimal import Decimal
from io import (
import json
import os
import sys
import time
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import IS64
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.json import ujson_dumps
def test_frame_nonprintable_bytes(self):

    class BinaryThing:

        def __init__(self, hexed) -> None:
            self.hexed = hexed
            self.binary = bytes.fromhex(hexed)

        def __str__(self) -> str:
            return self.hexed
    hexed = '574b4454ba8c5eb4f98a8f45'
    binthing = BinaryThing(hexed)
    df_printable = DataFrame({'A': [binthing.hexed]})
    assert df_printable.to_json() == f'{{"A":{{"0":"{hexed}"}}}}'
    df_nonprintable = DataFrame({'A': [binthing]})
    msg = 'Unsupported UTF-8 sequence length when encoding string'
    with pytest.raises(OverflowError, match=msg):
        df_nonprintable.to_json()
    df_mixed = DataFrame({'A': [binthing], 'B': [1]}, columns=['A', 'B'])
    with pytest.raises(OverflowError, match=msg):
        df_mixed.to_json()
    result = df_nonprintable.to_json(default_handler=str)
    expected = f'{{"A":{{"0":"{hexed}"}}}}'
    assert result == expected
    assert df_mixed.to_json(default_handler=str) == f'{{"A":{{"0":"{hexed}"}},"B":{{"0":1}}}}'