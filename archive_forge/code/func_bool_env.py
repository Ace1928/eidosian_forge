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
def bool_env(name, default=None):
    value = os.environ.get(name.upper())
    if not value:
        return default
    value = value.lower()
    if value in {'1', 'true', 'on', 'yes', 'y'}:
        return True
    elif value in {'0', 'false', 'off', 'no', 'n'}:
        return False
    else:
        raise ValueError('{}={} is not parsable as boolean'.format(name.upper(), value))