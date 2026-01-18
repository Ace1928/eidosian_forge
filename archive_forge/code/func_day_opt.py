from datetime import datetime
import pytest
from pandas._libs.tslibs.ccalendar import (
import pandas._libs.tslibs.offsets as liboffsets
from pandas._libs.tslibs.offsets import roll_qtrday
from pandas import Timestamp
@pytest.fixture(params=['start', 'end', 'business_start', 'business_end'])
def day_opt(request):
    return request.param