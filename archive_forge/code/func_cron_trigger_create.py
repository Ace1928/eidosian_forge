import os
import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
def cron_trigger_create(self, name, wf_name, wf_input, pattern=None, count=None, first_time=None, admin=True):
    optional_params = ''
    if pattern:
        optional_params += ' --pattern "{}"'.format(pattern)
    if count:
        optional_params += ' --count {}'.format(count)
    if first_time:
        optional_params += ' --first-time "{}"'.format(first_time)
    trigger = self.mistral_cli(admin, 'cron-trigger-create', params='{} {} {} {}'.format(name, wf_name, wf_input, optional_params))
    self.addCleanup(self.mistral_cli, admin, 'cron-trigger-delete', params=name)
    return trigger