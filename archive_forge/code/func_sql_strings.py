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
@pytest.fixture
def sql_strings():
    return {'read_parameters': {'sqlite': 'SELECT * FROM iris WHERE Name=? AND SepalLength=?', 'mysql': 'SELECT * FROM iris WHERE `Name`=%s AND `SepalLength`=%s', 'postgresql': 'SELECT * FROM iris WHERE "Name"=%s AND "SepalLength"=%s'}, 'read_named_parameters': {'sqlite': '\n                SELECT * FROM iris WHERE Name=:name AND SepalLength=:length\n                ', 'mysql': '\n                SELECT * FROM iris WHERE\n                `Name`=%(name)s AND `SepalLength`=%(length)s\n                ', 'postgresql': '\n                SELECT * FROM iris WHERE\n                "Name"=%(name)s AND "SepalLength"=%(length)s\n                '}, 'read_no_parameters_with_percent': {'sqlite': "SELECT * FROM iris WHERE Name LIKE '%'", 'mysql': "SELECT * FROM iris WHERE `Name` LIKE '%'", 'postgresql': 'SELECT * FROM iris WHERE "Name" LIKE \'%\''}}