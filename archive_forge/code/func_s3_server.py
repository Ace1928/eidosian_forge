import functools
import os
import pathlib
import subprocess
import sys
import time
import urllib.request
import pytest
import hypothesis as h
from ..conftest import groups, defaults
from pyarrow import set_timezone_db_path
from pyarrow.util import find_free_port
@pytest.fixture(scope='session')
def s3_server(s3_connection, tmpdir_factory):

    @retry(attempts=5, delay=0.1, backoff=2)
    def minio_server_health_check(address):
        resp = urllib.request.urlopen(f'http://{address}/minio/health/cluster')
        assert resp.getcode() == 200
    tmpdir = tmpdir_factory.getbasetemp()
    host, port, access_key, secret_key = s3_connection
    address = '{}:{}'.format(host, port)
    env = os.environ.copy()
    env.update({'MINIO_ACCESS_KEY': access_key, 'MINIO_SECRET_KEY': secret_key})
    args = ['minio', '--compat', 'server', '--quiet', '--address', address, tmpdir]
    proc = None
    try:
        proc = subprocess.Popen(args, env=env)
    except OSError:
        pytest.skip('`minio` command cannot be located')
    else:
        minio_server_health_check(address)
        yield {'connection': s3_connection, 'process': proc, 'tempdir': tmpdir}
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()