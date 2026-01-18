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
def test_filesystem_equals():
    fs0 = LocalFileSystem()
    fs1 = LocalFileSystem()
    fs2 = _MockFileSystem()
    assert fs0.equals(fs0)
    assert fs0.equals(fs1)
    with pytest.raises(TypeError):
        fs0.equals('string')
    assert fs0 == fs0 == fs1
    assert fs0 != 4
    assert fs2 == fs2
    assert fs2 != _MockFileSystem()
    assert SubTreeFileSystem('/base', fs0) == SubTreeFileSystem('/base', fs0)
    assert SubTreeFileSystem('/base', fs0) != SubTreeFileSystem('/base', fs2)
    assert SubTreeFileSystem('/base', fs0) != SubTreeFileSystem('/other', fs0)