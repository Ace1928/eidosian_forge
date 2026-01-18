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
def name_contains(test_name, vals):
    """Determines if any string in vals is a substring of test_name.

    Args:
        test_name: (string) String to determine if contains substrings.
        vals: (list of strings) List of substrings to test for.

    Returns:
        True if a substring in vals is in test_name, else False.
    """
    return any((val in test_name for val in vals))