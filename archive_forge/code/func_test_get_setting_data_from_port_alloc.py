from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@ddt.data(True, False)
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_get_setting_data_from_port_alloc(self, enable_cache, mock_get_elem_assoc_cls):
    self.netutils._enable_cache = enable_cache
    sd_object = mock.MagicMock()
    mock_port = mock.MagicMock(InstanceID=mock.sentinel.InstanceID)
    mock_get_elem_assoc_cls.return_value = [sd_object]
    cache = {}
    result = self.netutils._get_setting_data_from_port_alloc(mock_port, cache, mock.sentinel.data_class)
    mock_get_elem_assoc_cls.assert_called_once_with(self.netutils._conn, mock.sentinel.data_class, element_instance_id=mock.sentinel.InstanceID)
    self.assertEqual(sd_object, result)
    expected_cache = {mock.sentinel.InstanceID: sd_object} if enable_cache else {}
    self.assertEqual(expected_cache, cache)