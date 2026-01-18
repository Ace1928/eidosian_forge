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
@pytest.fixture(scope='module')
def s3_base():
    with ensure_safe_environment_variables():
        os.environ['AWS_ACCESS_KEY_ID'] = 'foobar_key'
        os.environ['AWS_SECRET_ACCESS_KEY'] = 'foobar_secret'
        os.environ['AWS_SHARED_CREDENTIALS_FILE'] = ''
        os.environ['AWS_CONFIG_FILE'] = ''
        proc = subprocess.Popen(shlex.split('moto_server s3 -p 5555'), stdout=subprocess.DEVNULL)
        timeout = 8
        while True:
            try:
                r = requests.get(endpoint_uri)
                if r.ok:
                    break
            except Exception:
                pass
            timeout -= 0.1
            time.sleep(0.1)
            assert timeout > 0, 'Timed out waiting for moto server'
        yield
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            if sys.platform == 'win32':
                subprocess.call(f'TASKKILL /F /PID {proc.pid} /T')