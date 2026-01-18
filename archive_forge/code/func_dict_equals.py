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
def dict_equals(dict1, dict2):
    """Check whether two dictionaries are equal and raise an ``AssertionError`` if they aren't."""
    for key1, key2 in itertools.zip_longest(sorted(dict1), sorted(dict2)):
        value_equals(key1, key2)
        value_equals(dict1[key1], dict2[key2])