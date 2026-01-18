import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_event_tr_create_missing_argument(self):
    wf = self.workflow_create(self.wf_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'event-trigger-create', params='tr %s exchange topic' % wf[0]['ID'])