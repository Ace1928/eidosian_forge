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
def flavor(conn_name):
    if 'postgresql' in conn_name:
        return 'postgresql'
    elif 'sqlite' in conn_name:
        return 'sqlite'
    elif 'mysql' in conn_name:
        return 'mysql'
    raise ValueError(f'unsupported connection: {conn_name}')