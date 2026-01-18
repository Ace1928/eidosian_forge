import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import tasks
def test_list_tasks_for_owner(self):
    filters = {'filters': {'owner': _OWNER_ID}}
    tasks = self.controller.list(**filters)
    self.assertEqual(_OWNED_TASK_ID, tasks[0].id)