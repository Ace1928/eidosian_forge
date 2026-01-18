from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_create_with_instance_obj(self):
    create_mock = mock.Mock()
    self.backups._create = create_mock
    args = {'name': 'test_backup', 'instance': self.instance_with_id.id, 'incremental': False}
    body = {'backup': args}
    self.backups.create('test_backup', self.instance_with_id)
    create_mock.assert_called_with('/backups', body, 'backup')