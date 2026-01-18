import numpy as np
import pytest
from pandas import (
from pandas.io.formats.format import EngFormatter
@pytest.fixture(autouse=True)
def reset_float_format():
    yield
    reset_option('display.float_format')