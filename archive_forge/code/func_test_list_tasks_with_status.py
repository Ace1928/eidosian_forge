import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import tasks
def test_list_tasks_with_status(self):
    filters = {'filters': {'status': 'processing'}}
    tasks = self.controller.list(**filters)
    self.assertEqual(_OWNED_TASK_ID, tasks[0].id)