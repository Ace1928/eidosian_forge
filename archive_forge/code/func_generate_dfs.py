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
def generate_dfs():
    df = pandas.DataFrame({'col1': [0, 1, 2, 3], 'col2': [4, 5, 6, 7], 'col3': [8, 9, 10, 11], 'col4': [12, 13, 14, 15], 'col5': [0, 0, 0, 0]})
    df2 = pandas.DataFrame({'col1': [0, 1, 2, 3], 'col2': [4, 5, 6, 7], 'col3': [8, 9, 10, 11], 'col6': [12, 13, 14, 15], 'col7': [0, 0, 0, 0]})
    return (df, df2)