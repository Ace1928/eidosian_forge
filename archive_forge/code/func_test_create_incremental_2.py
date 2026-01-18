from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_create_incremental_2(self):
    create_mock = mock.Mock()
    self.backups._create = create_mock
    args = {'name': 'test_backup', 'instance': '1', 'incremental': True}
    body = {'backup': args}
    self.backups.create(**args)
    create_mock.assert_called_with('/backups', body, 'backup')