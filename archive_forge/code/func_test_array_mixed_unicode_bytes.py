import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
def test_array_mixed_unicode_bytes():
    check_array_mixed_unicode_bytes(pa.binary(), pa.string())
    check_array_mixed_unicode_bytes(pa.large_binary(), pa.large_string())