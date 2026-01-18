import itertools
import numpy as np
import pytest
from pandas import (
@pytest.fixture(params=create_series())
def series_data(request):
    return request.param