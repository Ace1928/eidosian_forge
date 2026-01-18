import errno
import os
import shutil
import tempfile
import warnings
from unittest.mock import patch
from tempfile import TemporaryDirectory
from testpath import assert_isdir, assert_isfile, modified_env
from IPython import paths
from IPython.testing.decorators import skip_win32
@skip_win32
def test_get_ipython_dir_8():
    """test_get_ipython_dir_8, test / home directory"""
    if not os.access('/', os.W_OK):
        return
    with patch.object(paths, '_writable_dir', lambda path: bool(path)), patch.object(paths, 'get_xdg_dir', return_value=None), modified_env({'IPYTHON_DIR': None, 'IPYTHONDIR': None, 'HOME': '/'}):
        assert paths.get_ipython_dir() == '/.ipython'