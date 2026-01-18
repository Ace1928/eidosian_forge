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
def cloudpickle():
    cp = pytest.importorskip('cloudpickle')
    if 'HIGHEST_PROTOCOL' not in cp.__dict__:
        cp.HIGHEST_PROTOCOL = cp.DEFAULT_PROTOCOL
    return cp