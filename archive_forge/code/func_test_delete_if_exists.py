from unittest import mock
import os_brick.privileged as privsep_brick
import os_brick.privileged.rbd as privsep_rbd
from os_brick.tests import base
@mock.patch.object(privsep_rbd, 'get_rbd_class')
@mock.patch('oslo_utils.fileutils.delete_if_exists')
def test_delete_if_exists(self, mock_delete, mock_get_class):
    res = privsep_rbd.delete_if_exists(mock.sentinel.path)
    mock_get_class.assert_not_called()
    mock_delete.assert_called_once_with(mock.sentinel.path)
    self.assertIs(mock_delete.return_value, res)