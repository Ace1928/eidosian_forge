from datetime import (
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
@pytest.fixture(params=[True, False])
def ignore_na(request):
    """ignore_na keyword argument for ewm"""
    return request.param