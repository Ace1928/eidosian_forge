import collections
from functools import partial
import string
import subprocess
import sys
import textwrap
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
from pandas.core import ops
import pandas.core.common as com
from pandas.util.version import Version
def test_non_bool_array_with_na(self):
    arr = np.array(['A', 'B', np.nan], dtype=object)
    assert not com.is_bool_indexer(arr)