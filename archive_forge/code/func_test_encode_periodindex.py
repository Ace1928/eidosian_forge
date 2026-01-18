import calendar
import datetime
import decimal
import json
import locale
import math
import re
import time
import dateutil
import numpy as np
import pytest
import pytz
import pandas._libs.json as ujson
from pandas.compat import IS64
from pandas import (
import pandas._testing as tm
def test_encode_periodindex(self):
    p = PeriodIndex(['2022-04-06', '2022-04-07'], freq='D')
    df = DataFrame(index=p)
    assert df.to_json() == '{}'