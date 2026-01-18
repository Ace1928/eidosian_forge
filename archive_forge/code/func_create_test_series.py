import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def create_test_series(vals, sort=False, **kwargs):
    if isinstance(vals, dict):
        modin_series = pd.Series(vals[next(iter(vals.keys()))], **kwargs)
        pandas_series = pandas.Series(vals[next(iter(vals.keys()))], **kwargs)
    else:
        modin_series = pd.Series(vals, **kwargs)
        pandas_series = pandas.Series(vals, **kwargs)
    if sort:
        modin_series = modin_series.sort_values().reset_index(drop=True)
        pandas_series = pandas_series.sort_values().reset_index(drop=True)
    return (modin_series, pandas_series)