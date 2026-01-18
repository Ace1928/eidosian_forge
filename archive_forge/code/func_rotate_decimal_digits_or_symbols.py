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
def rotate_decimal_digits_or_symbols(value):
    if value.dtype == object:
        return [x[-1] + x[:-1] for x in value]
    else:
        tens = value // 10
        ones = value % 10
        return tens + ones * 10