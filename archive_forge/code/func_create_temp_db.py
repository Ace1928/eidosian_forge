import sqlite3
from tempfile import NamedTemporaryFile
from unittest import SkipTest
import numpy as np
import pandas as pd
from holoviews.core.data import Dataset
from holoviews.core.data.ibis import IbisInterface
from holoviews.core.spaces import HoloMap
from .base import HeterogeneousColumnTests, InterfaceTests, ScalarColumnTests
def create_temp_db(df, name, index=False):
    with NamedTemporaryFile(delete=False) as my_file:
        filename = my_file.name
    con = sqlite3.Connection(filename)
    df.to_sql(name, con, index=index)
    return sqlite.connect(filename)