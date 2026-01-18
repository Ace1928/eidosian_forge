from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
def open_input_stream(self, path):
    if 'notfound' in path:
        raise FileNotFoundError(path)
    data = '{0}:input_stream'.format(path).encode('utf8')
    return pa.BufferReader(data)