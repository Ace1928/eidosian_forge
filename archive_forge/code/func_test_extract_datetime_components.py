from collections import namedtuple
import datetime
import decimal
from functools import lru_cache, partial
import inspect
import itertools
import math
import os
import pytest
import random
import sys
import textwrap
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import ArrowNotImplementedError
from pyarrow.tests import util
@pytest.mark.pandas
def test_extract_datetime_components():
    timestamps = ['1970-01-01T00:00:59.123456789', '2000-02-29T23:23:23.999999999', '2033-05-18T03:33:20.000000000', '2020-01-01T01:05:05.001', '2019-12-31T02:10:10.002', '2019-12-30T03:15:15.003', '2009-12-31T04:20:20.004132', '2010-01-01T05:25:25.005321', '2010-01-03T06:30:30.006163', '2010-01-04T07:35:35.0', '2006-01-01T08:40:40.0', '2005-12-31T09:45:45.0', '2008-12-28T00:00:00.0', '2008-12-29T00:00:00.0', '2012-01-01T01:02:03.0']
    timezones = ['UTC', 'US/Central', 'Asia/Kolkata', 'Etc/GMT-4', 'Etc/GMT+4', 'Australia/Broken_Hill']
    _check_datetime_components(timestamps)
    if sys.platform == 'win32' and (not util.windows_has_tzdata()):
        pytest.skip('Timezone database is not installed on Windows')
    else:
        for timezone in timezones:
            _check_datetime_components(timestamps, timezone)