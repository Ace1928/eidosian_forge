from __future__ import annotations
import contextlib
import datetime
import numpy as np
import pandas as pd
import pytest
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_210
from dask.dataframe.utils import assert_eq
def get_shifted_sum_arg(overlap):
    return overlap.seconds - 1 if isinstance(overlap, datetime.timedelta) else overlap