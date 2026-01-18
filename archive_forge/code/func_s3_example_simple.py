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
@pytest.fixture
def s3_example_simple(s3_server):
    from pyarrow.fs import FileSystem
    host, port, access_key, secret_key = s3_server['connection']
    uri = 's3://{}:{}@mybucket/data.parquet?scheme=http&endpoint_override={}:{}&allow_bucket_creation=True'.format(access_key, secret_key, host, port)
    fs, path = FileSystem.from_uri(uri)
    fs.create_dir('mybucket')
    table = pa.table({'a': [1, 2, 3]})
    with fs.open_output_stream('mybucket/data.parquet') as out:
        pq.write_table(table, out)
    return (table, path, fs, uri, host, port, access_key, secret_key)