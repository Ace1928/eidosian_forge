from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
def test_nat_pinned_docstrings():
    assert NaT.ctime.__doc__ == Timestamp.ctime.__doc__