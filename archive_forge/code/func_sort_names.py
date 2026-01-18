import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.fixture(params=['inner', ['outer'], 'A', [('B', 5)], ['inner', 'outer'], [('B', 5), 'outer'], ['A', ('B', 5)], ['inner', 'outer']])
def sort_names(request):
    return request.param