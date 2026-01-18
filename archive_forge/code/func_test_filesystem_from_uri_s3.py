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
@pytest.mark.s3
def test_filesystem_from_uri_s3(s3_server):
    from pyarrow.fs import S3FileSystem
    host, port, access_key, secret_key = s3_server['connection']
    uri = 's3://{}:{}@mybucket/foo/bar?scheme=http&endpoint_override={}:{}&allow_bucket_creation=True'.format(access_key, secret_key, host, port)
    fs, path = FileSystem.from_uri(uri)
    assert isinstance(fs, S3FileSystem)
    assert path == 'mybucket/foo/bar'
    fs.create_dir(path)
    [info] = fs.get_file_info([path])
    assert info.path == path
    assert info.type == FileType.Directory