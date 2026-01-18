from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.fixture
def education_df():
    return DataFrame({'gender': ['male', 'male', 'female', 'male', 'female', 'male'], 'education': ['low', 'medium', 'high', 'low', 'high', 'low'], 'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']})