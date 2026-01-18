from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_add_virt_feature(self):
    self._test_virt_method('AddFeatureSettings', 3, 'add_virt_feature', True, mock.sentinel.vm_path, [mock.sentinel.res_data])