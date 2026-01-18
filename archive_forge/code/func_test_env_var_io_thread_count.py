import os
import subprocess
import sys
import pytest
import pyarrow as pa
from pyarrow.lib import ArrowInvalid
def test_env_var_io_thread_count():
    code = 'if 1:\n        import pyarrow as pa\n        print(pa.io_thread_count())\n        '

    def run_with_env_var(env_var):
        env = os.environ.copy()
        env['ARROW_IO_THREADS'] = env_var
        res = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True)
        res.check_returncode()
        return (res.stdout.decode(), res.stderr.decode())
    out, err = run_with_env_var('17')
    assert out.strip() == '17'
    assert err == ''
    for v in ('-1', 'z'):
        out, err = run_with_env_var(v)
        assert out.strip() == '8'
        assert 'ARROW_IO_THREADS does not contain a valid number of threads' in err.strip()