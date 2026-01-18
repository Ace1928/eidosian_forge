from __future__ import annotations
import io
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
import pytest
from fsspec.compression import compr
from fsspec.core import get_fs_token_paths, open_files
from s3fs import S3FileSystem as DaskS3FileSystem
from tlz import concat, valmap
from dask import compute
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
@contextmanager
def s3_context(bucket=test_bucket_name, files=files):
    client = boto3.client('s3', endpoint_url=endpoint_uri)
    client.create_bucket(Bucket=bucket, ACL='public-read-write')
    for f, data in files.items():
        client.put_object(Bucket=bucket, Key=f, Body=data)
    fs = s3fs.S3FileSystem(anon=True, client_kwargs={'endpoint_url': 'http://127.0.0.1:5555/'})
    s3fs.S3FileSystem.clear_instance_cache()
    fs.invalidate_cache()
    try:
        yield fs
    finally:
        fs.rm(bucket, recursive=True)