import numpy as np
import pytest
from pandas._libs import (
from pandas.compat import IS64
from pandas import Index
import pandas._testing as tm
def test_cache_readonly_preserve_docstrings():
    assert Index.hasnans.__doc__ is not None