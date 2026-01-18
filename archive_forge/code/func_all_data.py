import numpy as np
import pytest
import pandas as pd
from pandas.core.arrays.floating import (
@pytest.fixture(params=['data', 'data_missing'])
def all_data(request, data, data_missing):
    """Parametrized fixture returning 'data' or 'data_missing' float arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == 'data':
        return data
    elif request.param == 'data_missing':
        return data_missing