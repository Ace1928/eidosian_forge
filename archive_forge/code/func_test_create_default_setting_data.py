from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_create_default_setting_data(self):
    result = self.netutils._create_default_setting_data('FakeClass')
    fake_class = self.netutils._conn.FakeClass
    self.assertEqual(fake_class.new.return_value, result)
    fake_class.new.assert_called_once_with()