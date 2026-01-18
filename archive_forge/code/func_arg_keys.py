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
def arg_keys(arg_name, keys):
    """Appends arg_name to the front of all values in keys.

    Args:
        arg_name: (string) String containing argument name.
        keys: (list of strings) Possible inputs of argument.

    Returns:
        List of strings with arg_name append to front of keys.
    """
    return ['{0}_{1}'.format(arg_name, key) for key in keys]