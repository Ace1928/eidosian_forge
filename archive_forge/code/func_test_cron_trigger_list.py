import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_cron_trigger_list(self):
    triggers = self.parser.listing(self.mistral('cron-trigger-list'))
    self.assertTableStruct(triggers, ['Name', 'Workflow', 'Pattern', 'Next execution time', 'Remaining executions', 'Created at', 'Updated at'])