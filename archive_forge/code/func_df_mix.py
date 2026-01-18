import io
import numpy as np
import pytest
from pandas import (
@pytest.fixture
def df_mix():
    return DataFrame([[-3], [1], [2]])