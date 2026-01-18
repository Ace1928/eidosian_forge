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
def sort_data(data):
    """Sort the passed sequence."""
    if isinstance(data, (pandas.DataFrame, pd.DataFrame)):
        return data.sort_values(data.columns.to_list(), ignore_index=True)
    elif isinstance(data, (pandas.Series, pd.Series)):
        return data.sort_values()
    else:
        return np.sort(data)