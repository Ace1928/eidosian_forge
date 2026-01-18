import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
def test__device_path_from_symlink_file_handle(self):
    """Get device name for a file handle (eg: RBD)."""
    handle = io.StringIO()
    res = utils._device_path_from_symlink(handle)
    self.assertEqual(handle, res)