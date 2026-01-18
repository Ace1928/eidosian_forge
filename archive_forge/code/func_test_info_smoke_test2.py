from io import StringIO
import re
from string import ascii_uppercase
import sys
import textwrap
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
def test_info_smoke_test2(float_frame):
    buf = StringIO()
    float_frame.reindex(columns=['A']).info(verbose=False, buf=buf)
    float_frame.reindex(columns=['A', 'B']).info(verbose=False, buf=buf)
    DataFrame().info(buf=buf)