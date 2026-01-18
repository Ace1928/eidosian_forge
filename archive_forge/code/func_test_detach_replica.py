from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def test_detach_replica(self):
    db_detach_mock = mock.Mock(return_value=None)
    self.instance.manager.edit = db_detach_mock
    self.instance.id = 1
    self.instance.detach_replica()
    self.assertEqual(1, db_detach_mock.call_count)