from datetime import datetime
import numpy as np
import pytest
from pandas import (
@pytest.fixture(params=resample_methods)
def resample_method(request):
    """Fixture for parametrization of Grouper resample methods."""
    return request.param