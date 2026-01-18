from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
@contextlib.contextmanager
def check_to_pydatetime_deprecation(catch_deprecation_warnings: bool):
    if PANDAS_GE_210 and catch_deprecation_warnings:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='.*DatetimeProperties.to_pydatetime is deprecated', category=FutureWarning)
            yield
    else:
        yield