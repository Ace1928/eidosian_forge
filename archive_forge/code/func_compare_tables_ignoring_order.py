import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
def compare_tables_ignoring_order(t1, t2):
    df1 = t1.to_pandas().sort_values('b').reset_index(drop=True)
    df2 = t2.to_pandas().sort_values('b').reset_index(drop=True)
    assert df1.equals(df2)