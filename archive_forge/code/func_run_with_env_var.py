import os
import subprocess
import sys
import pytest
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
def run_with_env_var(env_var):
    env = os.environ.copy()
    env['ARROW_IO_THREADS'] = env_var
    res = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True)
    res.check_returncode()
    return (res.stdout.decode(), res.stderr.decode())