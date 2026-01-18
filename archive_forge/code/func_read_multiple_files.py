import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
def read_multiple_files(paths, columns=None, use_threads=True, **kwargs):
    dataset = pq.ParquetDataset(paths, **kwargs)
    return dataset.read(columns=columns, use_threads=use_threads)