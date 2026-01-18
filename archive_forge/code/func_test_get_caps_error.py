from unittest import mock
from oslotest import base
from oslo_privsep import capabilities
@mock.patch('oslo_privsep.capabilities._capget')
def test_get_caps_error(self, mock_capget):
    mock_capget.return_value = -1
    self.assertRaises(OSError, capabilities.get_caps)