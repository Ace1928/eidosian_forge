from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_add_virt_resource(self):
    self._test_virt_method('AddResourceSettings', 3, 'add_virt_resource', True, mock.sentinel.vm_path, [mock.sentinel.res_data])