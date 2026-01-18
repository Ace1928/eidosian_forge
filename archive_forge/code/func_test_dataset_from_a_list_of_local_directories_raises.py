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
def test_dataset_from_a_list_of_local_directories_raises(multisourcefs):
    msg = 'points to a directory, but only file paths are supported'
    with pytest.raises(IsADirectoryError, match=msg):
        ds.dataset(['/plain', '/schema', '/hive'], filesystem=multisourcefs)