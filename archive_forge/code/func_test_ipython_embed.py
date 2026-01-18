import os
import subprocess
import sys
from IPython.utils.tempdir import NamedFileInTemporaryDirectory
from IPython.testing.decorators import skip_win32
from IPython.testing import IPYTHON_TESTING_TIMEOUT_SCALE
import IPython
def test_ipython_embed():
    """test that `IPython.embed()` works"""
    with NamedFileInTemporaryDirectory('file_with_embed.py') as f:
        f.write(_sample_embed)
        f.flush()
        f.close()
        cmd = [sys.executable, f.name]
        env = os.environ.copy()
        env['IPY_TEST_SIMPLE_PROMPT'] = '1'
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(_exit)
        std = out.decode('UTF-8')
        assert p.returncode == 0
        assert '3 . 14' in std
        if os.name != 'nt':
            assert 'IPython' in std
        assert 'bye!' in std