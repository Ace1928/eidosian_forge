import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def single_level_multiindex():
    """single level MultiIndex"""
    return MultiIndex(levels=[['foo', 'bar', 'baz', 'qux']], codes=[[0, 1, 2, 3]], names=['first'])