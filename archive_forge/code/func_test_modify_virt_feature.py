from unittest import mock
import ddt
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import jobutils
def test_modify_virt_feature(self):
    self._test_virt_method('ModifyFeatureSettings', 3, 'modify_virt_feature', False, FeatureSettings=[mock.sentinel.res_data])