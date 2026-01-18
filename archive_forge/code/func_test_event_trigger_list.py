import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_event_trigger_list(self):
    triggers = self.parser.listing(self.mistral('event-trigger-list'))
    self.assertTableStruct(triggers, ['ID', 'Name', 'Workflow ID', 'Exchange', 'Topic', 'Event', 'Created at', 'Updated at'])