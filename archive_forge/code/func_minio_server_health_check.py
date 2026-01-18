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
@retry(attempts=5, delay=0.1, backoff=2)
def minio_server_health_check(address):
    resp = urllib.request.urlopen(f'http://{address}/minio/health/cluster')
    assert resp.getcode() == 200