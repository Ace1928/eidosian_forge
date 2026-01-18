from __future__ import annotations
import contextlib
from contextlib import closing
import csv
from datetime import (
from io import StringIO
from pathlib import Path
import sqlite3
from typing import TYPE_CHECKING
import uuid
import numpy as np
import pytest
from pandas._libs import lib
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.util.version import Version
from pandas.io import sql
from pandas.io.sql import (
def format_query(sql, *args):
    _formatters = {datetime: "'{}'".format, str: "'{}'".format, np.str_: "'{}'".format, bytes: "'{}'".format, float: '{:.8f}'.format, int: '{:d}'.format, type(None): lambda x: 'NULL', np.float64: '{:.10f}'.format, bool: "'{!s}'".format}
    processed_args = []
    for arg in args:
        if isinstance(arg, float) and isna(arg):
            arg = None
        formatter = _formatters[type(arg)]
        processed_args.append(formatter(arg))
    return sql % tuple(processed_args)