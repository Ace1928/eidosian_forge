import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_event_tr_delete_nonexistent_tr(self):
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'event-trigger-delete', params='789')