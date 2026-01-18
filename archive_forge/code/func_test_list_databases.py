from unittest import mock
import testtools
from troveclient import base
from troveclient.v1 import instances
def test_list_databases(self):
    db_list = ['database1', 'database2']
    self.instance.manager.databases = mock.Mock()
    self.instance.manager.databases.list = mock.Mock(return_value=db_list)
    self.assertEqual(db_list, self.instance.list_databases())