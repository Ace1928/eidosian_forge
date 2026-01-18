from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
@mock.patch.object(networkutils.NetworkUtils, '_bind_security_rules')
def test_create_security_rules(self, mock_bind, mock_get_elem_assoc_cls):
    m_port, m_acl = self._setup_security_rule_test(mock_get_elem_assoc_cls)
    fake_rule = mock.MagicMock()
    self.netutils.create_security_rules(self._FAKE_PORT_NAME, fake_rule)
    mock_bind.assert_called_once_with(m_port, fake_rule)