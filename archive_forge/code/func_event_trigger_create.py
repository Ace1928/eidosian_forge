import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def event_trigger_create(self, name, wf_id, exchange, topic, event, wf_input, admin=True):
    trigger = self.mistral_cli(admin, 'event-trigger-create', params=' '.join((name, wf_id, exchange, topic, event, wf_input)))
    ev_tr_id = self.get_field_value(trigger, 'ID')
    self.addCleanup(self.mistral_cli, admin, 'event-trigger-delete', params=ev_tr_id)
    return trigger