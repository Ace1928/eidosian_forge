import numpy as np
import pytest
from pandas import (
from pandas.core.groupby.base import (
@pytest.fixture(params=[True, False])
def as_index(request):
    return request.param