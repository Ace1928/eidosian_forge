from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
def test_get_element_associated_class_invalid_element(self):
    self.assertRaises(exceptions.WqlException, _wqlutils.get_element_associated_class, mock.sentinel.conn, mock.sentinel.class_name)