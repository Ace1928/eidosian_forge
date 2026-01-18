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
def check_file_leaks(func):
    """
    A decorator that ensures that no *newly* opened file handles are left
    after decorated function is finished.
    """
    if not TrackFileLeaks.get():
        return func

    @functools.wraps(func)
    def check(*a, **kw):
        fstart = _get_open_files()
        try:
            return func(*a, **kw)
        finally:
            leaks = []
            for item in _get_open_files():
                try:
                    fstart.remove(item)
                except ValueError:
                    if item[0].startswith('/proc/'):
                        continue
                    if re.search('/tmp/ray/session_.*/logs', item[0]):
                        continue
                    leaks.append(item)
            assert not leaks, f'Unexpected open handles left for: {', '.join((item[0] for item in leaks))}'
    return check