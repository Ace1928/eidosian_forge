import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
def test_no_default_pickle():
    obj = tm.round_trip_pickle(lib.no_default)
    assert obj is lib.no_default