import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_executions_list(self):
    executions = self.parser.listing(self.mistral('execution-list'))
    self.assertTableStruct(executions, ['ID', 'Workflow name', 'Workflow ID', 'State', 'Created at', 'Updated at'])