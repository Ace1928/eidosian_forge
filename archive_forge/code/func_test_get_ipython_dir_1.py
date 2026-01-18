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
def test_get_ipython_dir_1():
    """test_get_ipython_dir_1, Testcase to see if we can call get_ipython_dir without Exceptions."""
    env_ipdir = os.path.join('someplace', '.ipython')
    with patch.object(paths, '_writable_dir', return_value=True), modified_env({'IPYTHONDIR': env_ipdir}):
        ipdir = paths.get_ipython_dir()
    assert ipdir == env_ipdir