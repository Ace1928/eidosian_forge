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
def test_get_ipython_dir_2():
    """test_get_ipython_dir_2, Testcase to see if we can call get_ipython_dir without Exceptions."""
    with patch_get_home_dir('someplace'), patch.object(paths, 'get_xdg_dir', return_value=None), patch.object(paths, '_writable_dir', return_value=True), patch('os.name', 'posix'), modified_env({'IPYTHON_DIR': None, 'IPYTHONDIR': None, 'XDG_CONFIG_HOME': None}):
        ipdir = paths.get_ipython_dir()
    assert ipdir == os.path.join('someplace', '.ipython')