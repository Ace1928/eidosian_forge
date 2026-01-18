import numpy as np
import pytest
from pandas import (
@pytest.fixture
def series_empty():
    return Series(dtype=object)