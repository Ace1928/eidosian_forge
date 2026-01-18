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
def df_equals_with_non_stable_indices(df1, df2):
    """Assert equality of two frames regardless of the index order for equal values."""
    df1, df2 = map(try_cast_to_pandas, (df1, df2))
    np.testing.assert_array_equal(df1.values, df2.values)
    sorted1, sorted2 = map(sort_index_for_equal_values, (df1, df2))
    df_equals(sorted1, sorted2)