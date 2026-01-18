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
def test_filesystem_pickling(fs, pickle_module):
    if fs.type_name.split('::')[-1] == 'mock':
        pytest.xfail(reason='MockFileSystem is not serializable')
    serialized = pickle_module.dumps(fs)
    restored = pickle_module.loads(serialized)
    assert isinstance(restored, FileSystem)
    assert restored.equals(fs)