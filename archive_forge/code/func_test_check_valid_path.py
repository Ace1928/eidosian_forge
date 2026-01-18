from unittest import mock
import os_brick.privileged as privsep_brick
import os_brick.privileged.rbd as privsep_rbd
from os_brick.tests import base
@mock.patch.object(privsep_rbd, 'get_rbd_class')
@mock.patch.object(privsep_rbd, 'open')
@mock.patch.object(privsep_rbd, 'RBDConnector')
def test_check_valid_path(self, mock_connector, mock_open, mock_get_class):
    res = privsep_rbd.check_valid_path(mock.sentinel.path)
    mock_get_class.assert_called_once_with()
    mock_open.assert_called_once_with(mock.sentinel.path, 'rb')
    mock_connector._check_valid_device.assert_called_once_with(mock_open.return_value.__enter__.return_value)
    self.assertEqual(mock_connector._check_valid_device.return_value, res)