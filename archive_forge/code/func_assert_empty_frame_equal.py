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
def assert_empty_frame_equal(df1, df2):
    """
    Test if df1 and df2 are empty.

    Parameters
    ----------
    df1 : pandas.DataFrame or pandas.Series
    df2 : pandas.DataFrame or pandas.Series

    Raises
    ------
    AssertionError
        If check fails.
    """
    if df1.empty and (not df2.empty) or (df2.empty and (not df1.empty)):
        assert False, "One of the passed frames is empty, when other isn't"
    elif df1.empty and df2.empty and (type(df1) is not type(df2)):
        assert False, f'Empty frames have different types: {type(df1)} != {type(df2)}'