from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
def test_check_compatible_with(self, arr1d):
    arr1d._check_compatible_with(arr1d[0])
    arr1d._check_compatible_with(arr1d[:1])
    arr1d._check_compatible_with(NaT)