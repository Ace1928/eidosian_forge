import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
def test_constructor_sequence_like(self):

    class DummyContainer(abc.Sequence):

        def __init__(self, lst) -> None:
            self._lst = lst

        def __getitem__(self, n):
            return self._lst.__getitem__(n)

        def __len__(self) -> int:
            return self._lst.__len__()
    lst_containers = [DummyContainer([1, 'a']), DummyContainer([2, 'b'])]
    columns = ['num', 'str']
    result = DataFrame(lst_containers, columns=columns)
    expected = DataFrame([[1, 'a'], [2, 'b']], columns=columns)
    tm.assert_frame_equal(result, expected, check_dtype=False)