from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_delete_500(self):
    resp = mock.Mock()
    resp.status_code = 500
    self.backups.api.client.delete = mock.Mock(return_value=(resp, None))
    self.assertRaises(Exception, self.backups.delete, 'backup1')