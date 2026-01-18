import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape.concat import concat
from pandas.core.reshape.merge import merge
@pytest.fixture
def household(self):
    household = DataFrame({'household_id': [1, 2, 3], 'male': [0, 1, 0], 'wealth': [196087.3, 316478.7, 294750]}, columns=['household_id', 'male', 'wealth']).set_index('household_id')
    return household