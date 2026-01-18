from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_port_security_acls_cached(self):
    mock_port = mock.MagicMock(ElementName=mock.sentinel.port_name)
    self.netutils._sg_acl_sds = {mock.sentinel.port_name: [mock.sentinel.fake_acl]}
    acls = self.netutils._get_port_security_acls(mock_port)
    self.assertEqual([mock.sentinel.fake_acl], acls)