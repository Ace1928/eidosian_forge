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
def get_file_info_selector(self, selector):
    if selector.base_dir != 'somedir':
        if selector.allow_not_found:
            return []
        else:
            raise FileNotFoundError(selector.base_dir)
    infos = [FileInfo('somedir/file1', FileType.File, size=123), FileInfo('somedir/subdir1', FileType.Directory)]
    if selector.recursive:
        infos += [FileInfo('somedir/subdir1/file2', FileType.File, size=456)]
    return infos