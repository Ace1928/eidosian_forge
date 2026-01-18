from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.fixture
def names_with_nulls_df(nulls_fixture):
    return DataFrame({'key': [1, 1, 1, 1], 'first_name': ['John', 'Anne', 'John', 'Beth'], 'middle_name': ['Smith', nulls_fixture, nulls_fixture, 'Louise']})