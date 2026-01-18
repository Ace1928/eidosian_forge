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
def s3_connection():
    host, port = ('localhost', find_free_port())
    access_key, secret_key = ('arrow', 'apachearrow')
    return (host, port, access_key, secret_key)