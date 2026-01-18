import itertools
import numpy as np
import pytest
from pandas import (
@pytest.fixture(params=[0, 2])
def min_periods(request):
    return request.param