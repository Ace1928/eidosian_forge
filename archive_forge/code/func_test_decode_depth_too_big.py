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
@pytest.mark.parametrize('too_big_char', ['[', '{'])
def test_decode_depth_too_big(self, too_big_char):
    with pytest.raises(ValueError, match='Reached object decoding depth limit'):
        ujson.ujson_loads(too_big_char * (1024 * 1024))