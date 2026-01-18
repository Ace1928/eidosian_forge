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
def test_hive_partitioning_nulls(tempdir):
    table = pa.table({'a': ['x', None, 'z'], 'b': ['x', 'y', None]})
    part = ds.HivePartitioning(pa.schema([pa.field('a', pa.string()), pa.field('b', pa.string())]), None, 'xyz')
    ds.write_dataset(table, tempdir, format='ipc', partitioning=part)
    _check_dataset_directories(tempdir, ['a=x/b=x', 'a=xyz/b=y', 'a=z/b=xyz'])