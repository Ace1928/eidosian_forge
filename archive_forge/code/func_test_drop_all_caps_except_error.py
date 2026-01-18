from unittest import mock
from oslotest import base
from oslo_privsep import capabilities
@mock.patch('oslo_privsep.capabilities._capset')
def test_drop_all_caps_except_error(self, mock_capset):
    mock_capset.return_value = -1
    self.assertRaises(OSError, capabilities.drop_all_caps_except, [0], [0], [0])