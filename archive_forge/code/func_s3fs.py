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
@pytest.fixture
def s3fs(request, s3_server):
    request.config.pyarrow.requires('s3')
    from pyarrow.fs import S3FileSystem
    host, port, access_key, secret_key = s3_server['connection']
    bucket = 'pyarrow-filesystem/'
    fs = S3FileSystem(access_key=access_key, secret_key=secret_key, endpoint_override='{}:{}'.format(host, port), scheme='http', allow_bucket_creation=True, allow_bucket_deletion=True)
    fs.create_dir(bucket)
    yield dict(fs=fs, pathfn=bucket.__add__, allow_move_dir=False, allow_append_to_file=False)
    fs.delete_dir(bucket)