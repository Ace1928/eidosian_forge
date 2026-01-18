import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def make_tz_aware(series, tz):
    """
    Make a datetime64 Series timezone-aware for the given tz
    """
    tz = pa.lib.string_to_tzinfo(tz)
    series = series.dt.tz_localize('utc').dt.tz_convert(tz)
    return series