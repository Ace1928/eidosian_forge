import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.fixture(params=[True, False])
def groupby_series(request):
    return request.param