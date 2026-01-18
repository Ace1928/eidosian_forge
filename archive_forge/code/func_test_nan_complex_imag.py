from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
def test_nan_complex_imag(self):
    nan1 = complex(1, float('nan'))
    nan2 = complex(1, float('nan'))
    other = complex(2, float('nan'))
    assert nan1 is not nan2
    table = ht.PyObjectHashTable()
    table.set_item(nan1, 42)
    assert table.get_item(nan2) == 42
    with pytest.raises(KeyError, match=None) as error:
        table.get_item(other)
    assert str(error.value) == str(other)