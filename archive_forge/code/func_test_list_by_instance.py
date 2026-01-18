from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_list_by_instance(self):
    page_mock = mock.Mock()
    self.backups._paginated = page_mock
    instance_id = 'fake_instance'
    self.backups.list(instance_id=instance_id)
    page_mock.assert_called_with('/backups', 'backups', None, None, {'instance_id': instance_id})