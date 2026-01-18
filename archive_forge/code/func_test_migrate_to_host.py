import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import management
def test_migrate_to_host(self):
    hostname = 'hostname2'
    self._mock_action()
    self.management.migrate(1, host=hostname)
    self.assertEqual(1, self.management._action.call_count)
    self.assertEqual({'migrate': {'host': hostname}}, self.body_)