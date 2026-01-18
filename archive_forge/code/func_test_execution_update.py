import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_execution_update(self):
    execution = self.execution_create(self.async_wf['Name'])
    exec_id = self.get_field_value(execution, 'ID')
    status = self.get_field_value(execution, 'State')
    self.assertEqual('RUNNING', status)
    execution = self.mistral_admin('execution-update', params='{0} -s PAUSED'.format(exec_id))
    updated_exec_id = self.get_field_value(execution, 'ID')
    status = self.get_field_value(execution, 'State')
    self.assertEqual(exec_id, updated_exec_id)
    self.assertEqual('PAUSED', status)
    execution = self.mistral_admin('execution-update', params='{0} -d "execution update test"'.format(exec_id))
    description = self.get_field_value(execution, 'Description')
    self.assertEqual('execution update test', description)