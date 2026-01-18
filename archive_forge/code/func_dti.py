from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
@pytest.fixture
def dti():
    return date_range(start=datetime(2005, 1, 1), end=datetime(2005, 1, 10), freq='Min')