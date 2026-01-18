import io
import numpy as np
import pytest
from pandas import (
@pytest.fixture
def df_pos():
    return DataFrame([[1], [2], [3]])