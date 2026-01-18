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
def gcs_server():
    port = find_free_port()
    env = os.environ.copy()
    args = [sys.executable, '-m', 'testbench', '--port', str(port)]
    proc = None
    try:
        import testbench
        proc = subprocess.Popen(args, env=env)
        if proc.poll() is not None:
            pytest.skip(f'Command {args} did not start server successfully!')
    except (ModuleNotFoundError, OSError) as e:
        pytest.skip(f'Command {args} failed to execute: {e}')
    else:
        yield {'connection': ('localhost', port), 'process': proc}
    finally:
        if proc is not None:
            proc.kill()
            proc.wait()