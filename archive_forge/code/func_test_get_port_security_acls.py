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
def test_get_port_security_acls(self, enable_cache, mock_get_elem_assoc_cls):
    self.netutils._enable_cache = enable_cache
    self.netutils._sg_acl_sds = {}
    mock_port = mock.MagicMock()
    mock_get_elem_assoc_cls.return_value = [mock.sentinel.fake_acl]
    acls = self.netutils._get_port_security_acls(mock_port)
    self.assertEqual([mock.sentinel.fake_acl], acls)
    expected_cache = {mock_port.ElementName: [mock.sentinel.fake_acl]} if enable_cache else {}
    self.assertEqual(expected_cache, self.netutils._sg_acl_sds)
    mock_get_elem_assoc_cls.assert_called_once_with(self.netutils._conn, self.netutils._PORT_EXT_ACL_SET_DATA, element_instance_id=mock_port.InstanceID)