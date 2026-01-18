from __future__ import annotations
import re
import warnings
from collections.abc import Iterable
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
import dask
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_200, PANDAS_GE_300, tm
from dask.dataframe.core import apply_and_enforce
from dask.dataframe.utils import (
from dask.local import get_sync
def custom_scheduler(*args, **kwargs):
    nonlocal using_custom_scheduler
    try:
        using_custom_scheduler = True
        return get_sync(*args, **kwargs)
    finally:
        using_custom_scheduler = False