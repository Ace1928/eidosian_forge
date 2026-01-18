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
def test_keyword_deprecation(sqlite_engine):
    conn = sqlite_engine
    msg = "Starting with pandas version 3.0 all arguments of to_sql except for the arguments 'name' and 'con' will be keyword-only."
    df = DataFrame([{'A': 1, 'B': 2, 'C': 3}, {'A': 1, 'B': 2, 'C': 3}])
    df.to_sql('example', conn)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        df.to_sql('example', conn, None, if_exists='replace')