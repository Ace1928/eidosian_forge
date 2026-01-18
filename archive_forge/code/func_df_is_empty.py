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
def df_is_empty(df):
    """Tests if df is empty.

    Args:
        df: (pandas or modin DataFrame) dataframe to test if empty.

    Returns:
        True if df is empty.
    """
    assert df.size == 0 and df.empty
    assert df.shape[0] == 0 or df.shape[1] == 0